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
    ) where

import Numeric.LinearAlgebra
import qualified Numeric.LinearAlgebra as LA
import Util (enumerate, sigmoid, sigmoid', matrixToRows)
import Data.List (transpose)
import System.Random (randomRIO)
import Control.Monad (replicateM)


data Layer = Layer
    { weights :: Matrix R
    , biases :: Matrix R
    , sz:: Int
    , activation :: R -> R
    }

-- 1. Try prev example
-- 2. If it doesn't work fix it
-- matrixToRows is broken
-- 
type Network = [Layer]

activate :: Layer -> Matrix R -> Matrix R
activate l x = cmap (activation l) (weights l LA.<> x + biases l)

initWeights :: Int -> Int -> IO (Matrix R)
initWeights n m = do
    values <- replicateM (n * m) (randomRIO (-0.1, 0.1))
    return $ (n><m) values

-- Initialize biases with small random values
initBiases :: Int -> IO (Matrix R)
initBiases n = do
    values <- replicateM n (randomRIO (-0.1, 0.1))
    return $ (n><1) values

initialize :: [Int] -> (R -> R) -> IO Network
initialize s f = mapM (\x -> do
    w <- initWeights 0 0
    b <- initBiases 0
    return Layer {weights = w, biases = b, sz = x, activation = f}) s

fit :: Network -> Matrix R -> IO Network
fit n x = mapM (\(layer, idx) -> do
    w <- initWeights (sz layer) (len idx)
    b <- initBiases $ sz layer
    return Layer {weights = w, biases = b, sz = sz layer, activation = activation layer}) (enumerate n)
    where
        len idx = if idx == 0 then fst (size x) else sz $ n!!(idx - 1)


forwardProp :: Network -> Matrix R -> [Matrix R]
forwardProp n x = scanl (flip activate) x n

-- output is from previous layer (AKA the input to this layer)
-- sigmoid'(input) is sigmoid(input) * (1 - sigmoid(input)) so we can substitute that in place
calculateDelta :: (Matrix R, Layer) -> Matrix R -> Matrix R
calculateDelta (output, nextL) deltaNext = (tr (weights nextL) LA.<> deltaNext) * cmap sigmoid' output

-- `outputs` comes right from forwardProp
-- Return a list of tuples with weights and biases for every layer
backProp :: Network -> [Matrix R] -> Matrix R -> [(Matrix R, Matrix R)]
backProp n outputs target = zip dW dB
    where
        outputError = (last outputs - target) * cmap sigmoid' (last outputs)
        reversedOutputs = tail $ reverse outputs
        reversedNextLayers = reverse $ tail n

        layersWithOutputs = zip reversedOutputs reversedNextLayers
        deltas = reverse $ scanl (flip calculateDelta) outputError layersWithOutputs

        dW = [delta LA.<> tr out | (delta, out) <- zip deltas (init outputs)]
        dB = deltas

-- Network -> [(weight matrix for the l:th layer, bias vector (n x 1 matrix) for the l:th layer
updateParams :: Network -> [(Matrix R, Matrix R)] -> Network
updateParams = zipWith (curry (\(l, (dW, dB)) -> Layer {weights=weights l - dW * 10, biases=biases l - dB * 10, sz=sz l, activation=activation l}))

-- Batch gradient descent
-- Train on every input and output vector and take the average of all the backpropped weights and apply them
trainStep :: Network -> Matrix R -> Matrix R -> Network
trainStep n x y = updateParams n totalParams
    where
        inputs = matrixToRows x
        outputs = matrixToRows y

        -- `gradients` is of type [[(Matrix R, Matrix R)]]. The outer array is for the training examples and the inner one is for the layers.
        gradients = [backProp n (forwardProp n input) output | (input, output) <- zip inputs outputs]
        totalParams = map f $ transpose gradients
        f gs = let len = fromIntegral (length gs) in foldl1 (\(accW, accB) (w, b) -> (accW + w / len, accB + b / len)) gs
        --params = map (\(a, b) -> (a / fromIntegral (length inputs), b / fromIntegral (length inputs))) totalParams

train :: Network -> Matrix R -> Matrix R -> Int -> Network
train n _ _ 0 = n
train n x y epochs = train (trainStep n x y) x y (epochs - 1)

predict :: Network -> Matrix R -> Matrix R
predict n x = last $ forwardProp n x
