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
    ) where

import Numeric.LinearAlgebra
import Util (enumerate)

data Layer = Layer 
    { weights :: Matrix R
    , biases :: Vector R
    , sz:: Int
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
sigmoidLayer n m = Layer {weights=initWeights n m, biases=initBiases n, activation=sigmoid, sz=n}

activate :: Layer -> Vector R -> Vector R
activate l x = cmap (activation l) (weights l #> x + biases l)

initialize :: [Int] -> (R -> R) -> Network
initialize s f = map (\x -> Layer {weights=initWeights 0 0, biases=fromList [], sz=x, activation=f}) s

-- TODO. The point of this function is to initialize the weight matrices for each layer to be suitable for the inputs
-- the feature number for a layer is the amount of outputs from the previous layer

-- feature vector
fit :: Network -> Vector R -> Network
fit n x = map (\(layer, idx) -> Layer {weights=initWeights (sz layer) (len idx), biases=initBiases (sz layer), sz=sz layer, activation=activation layer}) $ enumerate n
    where
        len idx = if idx == 0 then size x else sz $ n!!(idx - 1)

forwardProp :: Network -> Vector R -> Vector R
forwardProp n x = forwardPropHelper x (length n)
    where
        forwardPropHelper :: Vector R -> Int -> Vector R
        forwardPropHelper input s =    
            if s == 0 
                then input 
            else forwardPropHelper (activate (n!!(length n - s)) input) (s - 1)
