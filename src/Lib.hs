module Lib
    ( Network
    , Layer
    , sigmoid
    , sigmoid'
    , sigmoidLayer
    , initialize
    , fit
    , forwardProp
    , weights
    , biases
    , activate
    , backProp
    , calculateDelta
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

-- layer size -> number of features
sigmoidLayer :: Int -> Int -> Layer
sigmoidLayer n m = Layer {weights=initWeights n m, biases=initBiases n, activation=sigmoid, sz=n}

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
calculateDelta (output, nextL) deltaNext = (output * (1 - output)) * tr (weights nextL) LA.<> deltaNext

-- `outputs` comes right from forwardProp
-- Return a list of tuples with weights and biases for every layer
backProp :: Network -> [Matrix R] -> Matrix R -> [(Matrix R, Matrix R)]
backProp n outputs target = zip dWs dBs
    where
        outputError = cmap sigmoid' (target - last outputs)

        reversedOutputs = tail $ reverse outputs
        reversedNextLayers = reverse $ tail n
        
        layersWithOutputs = zip reversedOutputs reversedNextLayers
        deltas = reverse $ scanl (flip calculateDelta) outputError layersWithOutputs 

        dWs = [delta LA.<> tr out | (delta, out) <- zip (init deltas) (init outputs)]
        dBs = deltas
