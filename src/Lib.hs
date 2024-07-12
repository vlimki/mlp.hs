module Lib (fit, train, bgdTrainer, initialize, predict, trainXOR, loadMNIST, trainMNIST, convertToSoftmax) where

import Network
import Trainer
import Util
import Codec.Compression.GZip (decompress)
import Numeric.LinearAlgebra (R, Matrix, (><), fromLists)
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString as BS

-- Takes a 1x1 matrix (e.g [5.0]) and converts it to the softmax format (e.g [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
convertToSoftmax :: R -> [R]
convertToSoftmax val = [if x == val then 1.0 else 0.0 | x <- [1..10]]

-- Return (input, output)
loadMNIST :: IO (Matrix R, Matrix R)
loadMNIST = do
  trainData <- decompress <$> BL.readFile "./data/train-images-idx3-ubyte.gz"
  trainLabels <- decompress <$> BL.readFile "./data/train-labels-idx1-ubyte.gz"

  let images = Prelude.concat $ [getImage n (BL.toStrict trainData) | n <- [0..9]]
  let labels = [getLabel n (BL.toStrict trainLabels) | n <- [0..9]]

  let images' = (Prelude.length labels><784) images
  let labels' = fromLists $ map convertToSoftmax labels
  return (images', labels')

-- This code is inspired by https://github.com/ttesmer/haskell-mnist/blob/master/src/Network.hs.
-- The MNIST dataset is stored in binary, where the first 16 bytes are the header, and every image is 784 bytes (28x28 pixels)
-- Every byte is a value from 0-255, so we normalize the value to be anywhere between 0 and 1.
getImage :: Int -> BS.ByteString -> [R]
getImage n ds = [normalize $ BS.index ds (16 + n * 784 + s) | s <- [0..783]]
  where normalize x = fromIntegral x / 255

-- The label data is stored so that the first 8 bytes are the header of the file, and every label from there is just 1 bit.
getLabel :: Int -> BS.ByteString -> R
getLabel n s = fromIntegral $ BS.index s (n + 8)

-- Training the network to solve the MNIST problem.
-- The network architecture is input: 784 (or 28*28) pixel values -> layer 1: 512 neurons -> layer 2: 256 neurons -> output: 10 neurons
-- We're using the softmax function here since we're doing multi-class classification.
trainMNIST :: IO Network
trainMNIST = do
  (x, y) <- loadMNIST
  n1 <- initialize [512, 256, 10] [reluActivation, reluActivation, softmaxActivation]
  n2 <- fit (head $ matrixToRows x) n1

  let t = bgdTrainer 0.1 100
  let n3 = train t n2 x y

  mapM_ (\(input, output) -> putStrLn $ "Target: " ++ show output ++ ", Output: " ++ show (predict input n3)) $ zip (matrixToRows x) (matrixToRows y)

  let loss = eval n3 (matrixToRows x) (matrixToRows y)
  putStrLn $ "Loss: " ++ show loss

  return n3

-- Training the network to solve the XOR problem.
-- We have a network architecture that looks like this `input -> l1: 4 neurons -> output: 1 neuron`.
-- We're using the relu activation function for the hidden layers and the sigmoid activation function for the output layer.
trainXOR :: IO Network
trainXOR = do
  n1 <- initialize [4, 1] [reluActivation, sigmoidActivation]
  n2 <- fit (head $ matrixToRows xorInput) n1

  putStrLn "Initial weights:"
  printNet n2

  let t = bgdTrainer 0.1 10000
  let n3 = train t n2 xorInput xorOutput

  putStrLn "Final weights:"
  printNet n3

  putStrLn "Predictions:"
  mapM_ (\input -> putStrLn $ "Input: " ++ show input ++ ", Output: " ++ show (predict input n3)) (matrixToRows xorInput)

  let loss = eval n3 (matrixToRows xorInput) (matrixToRows xorOutput)
  putStrLn $ "Loss: " ++ show loss

  return n3
