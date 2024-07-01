module Lib
    (
        Network,
        Layer,
        sigmoid,
        sigmoid',
        sigmoidLayer,
        activate,
    ) where

import Numeric.LinearAlgebra

data Layer = Layer 
    { weights :: Matrix R
    , biases :: Vector R
    , activation :: R -> R
    }

type Network = [Layer]

initWeights :: Int -> Int -> Matrix R
initWeights n m = (n><m) $ replicate (n*m) 0.123

initBiases :: Int -> Vector R
initBiases n = fromList $ replicate n 0.123

sigmoid :: R -> R
sigmoid x = 1 / (1 + exp (-x))

sigmoid' :: R -> R
sigmoid' x = sigmoid x * (1 - sigmoid x)

-- layer size -> number of features
sigmoidLayer :: Int -> Int -> Layer
sigmoidLayer n m = Layer {weights=initWeights n m, biases=initBiases n, activation=sigmoid}

activate :: Layer -> Vector R -> Vector R
activate l x = cmap (activation l) (weights l #> x + biases l)

-- TODO. The point of this function is to initialize the weight matrices for each layer to be suitable for the inputs
-- the feature number for a layer is the amount of outputs from the previous layer
fit :: Network -> Matrix R -> Vector R -> Network
fit n x y = []
