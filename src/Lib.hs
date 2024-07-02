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
import Util (enumerate, sigmoid, sigmoid')

data Layer = Layer
    { weights :: Matrix R
    , biases :: Matrix R
    , sz:: Int
    , activation :: R -> R
    }

type Network = [Layer]

initWeights :: Int -> Int -> Matrix R
initWeights n m = (n><m) $ replicate (n*m) 0.123

initBiases :: Int -> Matrix R
initBiases n = (n><1) $ replicate n 0.123

activate :: Layer -> Matrix R -> Matrix R
activate l x = cmap (activation l) (weights l LA.<> x + biases l)

initialize :: [Int] -> (R -> R) -> Network
initialize s f = map (\x -> Layer {weights=initWeights 0 0, biases=initBiases 0, sz=x, activation=f}) s

-- net -> feature vector
fit :: Network -> Matrix R -> Network
fit n x = map (\(layer, idx) -> Layer {weights=initWeights (sz layer) (len idx), biases=initBiases (sz layer), sz=sz layer, activation=activation layer}) $ enumerate n
    where
        len idx = if idx == 0 then fst (size x) else sz $ n!!(idx - 1)

forwardProp :: Network -> Matrix R -> [Matrix R]
forwardProp n x = scanl (flip activate) x n

-- output is from previous layer (AKA the input to this layer)
-- sigmoid'(input) is sigmoid(input) * (1 - sigmoid(input)) so we can substitute that in place
calculateDelta :: (Matrix R, Layer) -> Matrix R -> Matrix R
calculateDelta (output, nextL) deltaNext = tr (weights nextL) LA.<> deltaNext * cmap sigmoid' output

-- `outputs` comes right from forwardProp
-- Return a list of tuples with weights and biases for every layer
backProp :: Network -> [Matrix R] -> Matrix R -> [(Matrix R, Matrix R)]
backProp n outputs target = zip dWs dBs
    where
        outputError = (last outputs - target) * cmap sigmoid' (last outputs)

        reversedOutputs = tail $ reverse outputs
        reversedNextLayers = reverse $ tail n

        layersWithOutputs = zip reversedOutputs reversedNextLayers
        deltas = reverse $ scanl (flip calculateDelta) outputError layersWithOutputs

        dWs = [delta LA.<> tr out | (delta, out) <- zip deltas outputs]
        dBs = deltas

-- Network -> [(weight matrix for the l:th layer, bias vector (n x 1 matrix) for the l:th layer
updateParams :: Network -> [(Matrix R, Matrix R)] -> Network
updateParams = zipWith (curry (\(l, (dW, dB)) -> Layer {weights=weights l - dW * 0.01, biases=biases l - dB * 0.01, sz=sz l, activation=activation l}))

trainStep :: Network -> Matrix R -> Matrix R -> Network
trainStep n x y = updateParams n $ backProp n (forwardProp n x) y

train :: Network -> Matrix R -> Matrix R -> Int -> Network
train n _ _ 0 = n
train n x y epochs = train (trainStep n x y) x y (epochs - 1)

predict :: Network -> Matrix R -> Matrix R
predict n x = last $ forwardProp n x
