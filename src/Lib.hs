module Lib
    ( Network
    , Layer
    , sigmoid
    , sigmoid'
    , initialize
    , fit
    , weights
    , biases
    , train
    , predict
    , updateParams
    , trainStep
    , forwardProp
    , backProp
    , eval
    , trainXOR
    , printNet
    ) where

import Numeric.LinearAlgebra
import qualified Numeric.LinearAlgebra as LA
import Util (enumerate, sigmoid, sigmoid', matrixToRows, xorInput, xorOutput, mse, relu, relu', clipGradients)
import Data.List (transpose)
import System.Random (randomRIO)
import Control.Monad (replicateM)


data Layer = Layer
    { weights :: Matrix R
    , biases :: Matrix R
    , sz:: Int
    , activation :: R -> R
    , activation' :: R -> R
    }

type Network = [Layer]

activate :: Layer -> Matrix R -> Matrix R
activate l x = cmap (activation l) (weights l LA.<> x + biases l)

xavierInit :: Int -> Int -> IO (Matrix R)
xavierInit n m = do
    let limit = sqrt (2.0 / fromIntegral n)
    values <- replicateM (n * m) (randomRIO (-limit, limit))
    return $ (n><m) values

initWeights :: Int -> Int -> Matrix R
initWeights n m = (n><m) $ replicate (n * m) 0.0
-- Initialize biases with small random values
initBiases :: Int -> IO (Matrix R)
initBiases n = do
    values <- replicateM n (randomRIO (-0.1, 0.1))
    return $ (n><1) values

initialize :: [Int] -> [R -> R] -> [R -> R] -> IO Network
initialize s f d = mapM (\(x, idx) -> do
    let w = initWeights 0 0
    b <- initBiases 0
    return Layer {weights = w, biases = b, sz = x, activation = f!!idx, activation' = d!!idx}) $ enumerate s


fit :: Network -> Matrix R -> IO Network
fit n x = mapM (\(layer, idx) -> do
    w <- xavierInit (sz layer) (len idx)
    b <- initBiases $ sz layer
    return Layer {weights = w, biases = b, sz = sz layer, activation = activation layer, activation' = activation' layer}) (enumerate n)
    where
        len idx = if idx == 0 then fst (size x) else sz $ n!!(idx - 1)


forwardProp :: Network -> Matrix R -> [Matrix R]
forwardProp n x = scanl (flip activate) x n

-- output is from previous layer (AKA the input to this layer)
-- sigmoid'(input) is sigmoid(input) * (1 - sigmoid(input)) so we can substitute that in place
calculateDelta :: (Layer, Matrix R, Layer) -> Matrix R -> Matrix R
calculateDelta (l, output, nextL) deltaNext = (tr (weights nextL) LA.<> deltaNext) * cmap (activation' l) output

-- `outputs` comes right from forwardProp
-- Return a list of tuples with weights and biases for every layer
backProp :: Network -> [Matrix R] -> Matrix R -> [(Matrix R, Matrix R)]
backProp n outputs target = zip dW dB
    where
        outputError = (last outputs - target) * cmap sigmoid' (last outputs)

        reversedLayers = tail $ reverse n
        reversedOutputs = tail $ reverse outputs
        reversedNextLayers = reverse $ tail n

        layersWithOutputs = zip3 reversedLayers reversedOutputs reversedNextLayers
        deltas = reverse $ scanl (flip calculateDelta) outputError layersWithOutputs

        dW = [delta LA.<> tr out | (delta, out) <- zip deltas (init outputs)]
        dB = deltas

-- Network -> [(weight matrix for the l:th layer, bias vector (n x 1 matrix) for the l:th layer
updateParams :: Network -> [(Matrix R, Matrix R)] -> Network
updateParams = zipWith (curry (\(l, (dW, dB)) -> Layer {weights=weights l - scale 0.1 dW, biases=biases l - scale 0.1 dB, sz=sz l, activation=activation l, activation' = activation' l}))

trainStep :: Network -> Matrix R -> Matrix R -> Network
trainStep n x y = updateParams n totalParams
    where
        inputs = matrixToRows x
        outputs = matrixToRows y

        gradients = [backProp n (forwardProp n input) output | (input, output) <- zip inputs outputs]
        totalParams = clipGradients $ map f $ transpose gradients
        f gs = let len = fromIntegral (length gs) in foldl1 (\(accW, accB) (w, b) -> (accW + w / len, accB + b / len)) gs

train :: Network -> Matrix R -> Matrix R -> Int -> Network
train n _ _ 0 = n
train n x y epochs = train (trainStep n x y) x y (epochs - 1)

predict :: Network -> Matrix R -> Matrix R
predict n x = last $ forwardProp n x

trainXOR :: IO Network
trainXOR = do
    n1 <- initialize [2, 2, 1] [relu, relu, sigmoid] [relu', relu', sigmoid']
    n2 <- fit n1 $ head $ matrixToRows xorInput

    putStrLn "Initial weights:"
    printNet n2

    let n3 = train n2 xorInput xorOutput 100000

    putStrLn "Final weights:"
    printNet n3

    putStrLn "Predictions:"
    mapM_ (\input -> putStrLn $ "Input: " ++ show input ++ ", Output: " ++ show (predict n3 input)) (matrixToRows xorInput)

    let loss = eval n3 (matrixToRows xorInput) (matrixToRows xorOutput)
    putStrLn $ "Loss: " ++ show loss

    return n3

printNet :: Network -> IO ()
printNet = mapM_ (\l -> do
    putStrLn $ "Weights: " ++ show (weights l)
    putStrLn $ "Biases: " ++ show (biases l))

eval :: Network -> [Matrix R] -> [Matrix R] -> R
eval n x = mse output
    where
        output = map (predict n) x
