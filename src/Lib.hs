{-# LANGUAGE BangPatterns #-}
module Lib (fit, train, bgdTrainer, initialize, predict, trainXOR, loadMNIST, trainMNIST, convertToSoftmax, saveParameters, loadParameters, evalMNIST) where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Network.Activation
import Network.Network
import Network.Trainer
import Numeric.LinearAlgebra (Matrix, R, fromLists, (><), size, maxIndex, flatten, scalar)
import Util
import System.IO
import Data.IORef
import Control.Monad (forM_)
import System.Random (randomRIO)
import Control.Monad (foldM, when)
import Control.DeepSeq (rnf)
import GHC.Stats (getRTSStatsEnabled, getRTSStats)
import Control.Parallel.Strategies (parListChunk, using, rdeepseq, rseq, parMap)

chunks :: Int -> [a] -> [[a]]
chunks _ [] = []
chunks n xs
  | n > 0     = take n xs : chunks n (drop n xs)
  | otherwise = error "Chunk size must be > 0"

-- Takes a 1x1 matrix (e.g [4.0]) and converts it to the softmax format (e.g [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
convertToSoftmax :: R -> [R]
convertToSoftmax val = [if x == val then 1.0 else 0.0 | x <- [1 .. 10]]

-- Does the opposite of `convertToSoftmax`. Takes a matrix like [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
softmaxToPredicted :: Matrix R -> Matrix R
softmaxToPredicted mat = scalar (fromIntegral $ maxIndex $ flatten mat)

loadTestMNIST :: IO (Matrix R, Matrix R)
loadTestMNIST = do
  testData <- decompress <$> BL.readFile "./data/mnist/t10k-images-idx3-ubyte.gz"
  testLabels <- decompress <$> BL.readFile "./data/mnist/t10k-labels-idx1-ubyte.gz"

  let !lbls = BL.toStrict testLabels
  let !imgs = BL.toStrict testData

  let !images =  Prelude.concat $ [getImage n imgs | n <- [0 .. 9999]]
  let !labels = [getLabel n lbls | n <- [0 .. 9999]]

  let !images' = (10000><784) images
  let !labels' = fromLists $! map convertToSoftmax labels

  images' `seq` labels' `seq` return (images',  labels')

evalMNIST :: IO Int
evalMNIST = do
  n <- loadParameters "./data/weights-mnist" "./data/biases-mnist" [relu, relu, softmax]
  (inputs, outputs) <- loadTestMNIST
  errors <- mapM (\(x, y) -> do
    let err = if softmaxToPredicted (predict x n) == softmaxToPredicted y then 0 else 1
    --putStrLn $ "----------\nOutput: " ++ show (softmaxToPredicted (predict x n)) ++ "\nTarget: " ++ show (softmaxToPredicted y) ++ "\nError: " ++ show err
    return err
    ) $ zip (matrixToRows inputs) (matrixToRows outputs)

  putStrLn $ "Correctly classified " ++ show (length errors - sum errors) ++ "/" ++ show (length errors) ++ " images."
  return $ sum errors

-- Return [(input, output)] in chunks
loadMNIST :: IO [(Matrix R, Matrix R)]
loadMNIST = do
  trainData <- decompress <$> BL.readFile "./data/mnist/train-images-idx3-ubyte.gz"
  trainLabels <- decompress <$> BL.readFile "./data/mnist/train-labels-idx1-ubyte.gz"

  let batchSize = 100

  let lbls = BL.toStrict trainLabels
  let imgs = BL.toStrict trainData

  let images = chunks batchSize $ [getImage n imgs | n <- [0 .. 9999]]
  let labels = chunks batchSize $ [getLabel n lbls | n <- [0 .. 9999]]

  let images' = map (\c -> (batchSize >< 784) $! concat c) images
  let labels' = map (\c -> fromLists $! map convertToSoftmax c) labels
  images' `seq` labels' `seq` return (zip images' labels')

-- This code is inspired by https://github.com/ttesmer/haskell-mnist/blob/master/src/Network.hs.
-- The MNIST dataset is stored in binary, where the first 16 bytes are the header, and every image is 784 bytes (28x28 pixels)
-- Every byte is a value from 0-255, so we normalize the value to be anywhere between 0 and 1.
getImage :: Int -> BS.ByteString -> [R]
getImage n ds = [normalize $ BS.index ds (16 + n * 784 + s) | s <- [0 .. 783]]
  where
    normalize x = fromIntegral x / 255

-- The label data is stored so that the first 8 bytes are the header of the file, and every label from there is just 1 bit.
getLabel :: Int -> BS.ByteString -> R
getLabel n s = fromIntegral $ BS.index s (n + 8)

-- Train with mini-batch SGD (Stochastic Gradient Descent)
-- Will have to implement a different trainer interface for this later.
-- Same as `parMap rseq (\x -> train t n (fst x) (snd x)) chunks`
trainChunksParallel :: Trainer p => p -> [Layer] -> [(Matrix R, Matrix R)] -> Network
--trainChunksParallel t n = parMap rseq (uncurry (train t n))
trainChunksParallel t = foldl (uncurry . train t)
        --foldl (\n' c -> train t n' (fst c) (snd c)) n cs `using` parListChunk 10 rseq
  --parMap rseq (\x -> let net = uncurry (train t n) x in net)

trainEpoch :: Trainer t => t -> [(Matrix R, Matrix R)] -> [Layer] -> IO Network
trainEpoch t cs n = do
  let trained = trainChunksParallel t n cs
  trained `seq` return trained

eval :: Network -> (Matrix R, Matrix R) -> IO ()
eval n (testInputs, testOutputs) = do
  errors <- mapM (\(x, y) -> do
    let err = if softmaxToPredicted (predict x n) == softmaxToPredicted y then 0 else 1 :: Int
    return err) $ zip (matrixToRows testInputs) (matrixToRows testOutputs)
  putStrLn $ "Correctly classified " ++ show (length errors - sum errors) ++ "/" ++ show (length errors) ++ " images."
  return ()

-- Training the network to solve the MNIST problem.
-- The network architecture is input: 784 (or 28*28) pixel values -> layer 1: 512 neurons -> layer 2: 256 neurons -> output: 10 neurons
-- We're using the softmax function here since we're doing multi-class classification.
trainMNIST :: IO Network
trainMNIST = do
  putStrLn "Loading dataset..."
  cs <- loadMNIST
  putStrLn "Dataset loaded."
  putStrLn "Initializing network..."
  n1 <- initialize [512, 256, 10] [relu, relu, softmax]
  n2 <- fit (head $ matrixToRows $ fst $ head cs) n1

  let t = bgdTrainer 0.01 1
  putStrLn "Network initialized."

  let es = [1..50 :: Int]
  putStrLn "Loading test dataset..."
  (testInputs, testOutputs) <- loadTestMNIST
  testInputs `seq` testOutputs `seq` putStrLn "Test dataset loaded."
  putStrLn "Training network..."
  n3 <- foldM (\net epoch -> do
    putStrLn $ "Epoch " ++ show epoch ++ "/" ++ show (last es)
    hFlush stdout
    let newLr = learningRate t / (1 + 0.001 * (fromIntegral (last es) - fromIntegral (epochs t)))
    shuffled <- shuffle cs
    net' <- trainEpoch t {learningRate = newLr} shuffled net
    when (even epoch) $ do
      Lib.eval net' (testInputs, testOutputs)
    seq (rnf net') (return net')) n2 es

  putStrLn "Network trained."

  --let loss = eval n3 (matrixToRows x) (matrixToRows y)
  --putStrLn $ "Loss: " ++ show loss

  putStrLn "Saving parameters..."
  saveParameters "./data/weights-mnist" "./data/biases-mnist" n3
  putStrLn "Parameters saved. Done."

  return n3

-- Initialize a network based on pre-loaded parameters.
-- How do we save weights in a file? Well, we could literally just save them using `show` and `read` since hmatrix supports them.
-- So we need to save an array of weight matrices and an array of bias matrices. We can get the format we want using `show $ map weights <network>`
-- Upon further inspection it looks like `hmatrix` already exports functions called `saveMatrix` and `readMatrix`. Well, too bad
loadParameters :: String -> String -> [Activation] -> IO [Layer]
loadParameters wPath bPath activations = do
  fWeights <- openFile wPath ReadMode
  fBiases <- openFile bPath ReadMode
  wData <- hGetContents fWeights
  bData <- hGetContents fBiases
  let ws = read wData :: [Matrix R]
  let bs = read bData :: [Matrix R]
  return $ zipWith3 (\w b a -> Layer{activation=a, weights=w, biases=b, sz=snd (size b)}) ws bs activations

saveParameters :: FilePath -> FilePath -> Network -> IO ()
saveParameters wPath bPath n = do
  let w = show $ map weights n
  let b = show $ map biases n

  writeFile wPath w
  writeFile bPath b
  return ()

-- Training the network to solve the XOR problem.
-- We have a network architecture that looks like this `input -> l1: 4 neurons -> output: 1 neuron`.
-- We're using the relu activation function for the hidden layers and the sigmoid activation function for the output layer.
trainXOR :: IO Network
trainXOR = do
  n1 <- initialize [4, 1] [relu, sigmoid]
  n2 <- fit (head $ matrixToRows xorInput) n1

  putStrLn "Initial weights:"
  printNet n2

  let t = bgdTrainer 0.1 5000
  let n3 = train t n2 xorInput xorOutput

  putStrLn "Final weights:"
  printNet n3

  putStrLn "Predictions:"
  mapM_ (\input -> putStrLn $ "Input: " ++ show input ++ ", Output: " ++ show (predict input n3)) (matrixToRows xorInput)

  let loss = Network.Network.eval n3 (matrixToRows xorInput) (matrixToRows xorOutput)
  putStrLn $ "Loss: " ++ show loss

  return n3

-- Shuffle function using Fisher-Yates algorithm
-- I'll admit, ChatGPT wrote 100% of this lmao
shuffle :: [a] -> IO [a]
shuffle xs = do
    ar <- newIORef xs
    let n = length xs
    forM_ [0..n-1] $ \i -> do
        j <- randomRIO (i, n-1)
        arr <- readIORef ar
        let elemI = arr !! i
            elemJ = arr !! j
            swappedArr = updateList arr i elemJ
            finalArr = updateList swappedArr j elemI
        writeIORef ar finalArr
    readIORef ar

updateList :: [a] -> Int -> a -> [a]
updateList lst idx newVal = take idx lst ++ [newVal] ++ drop (idx + 1) lst
