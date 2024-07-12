module Util (
  enumerate,
  sigmoid,
  sigmoid',
  mse,
  matrixToRows,
  xorInput,
  xorOutput,
  relu,
  relu',
  clipGradients,
  softmax, 
  softmax'
)
where

import Data.Bifunctor
import Numeric.LinearAlgebra

-- For iterating arrays with indices
enumerate :: [a] -> [(a, Int)]
enumerate a = zip a [0 .. length a]

-- The sigmoid function.
sigmoid :: Matrix R -> Matrix R
sigmoid = cmap (\x -> 1 / (1 + exp (-x)))

-- The sigmoid derivative is defined as sigmoid(x) * (1 - sigmoid(x)).
-- Here we just assume the argument to the function has already been "sigmoided" - as it is when calling this function in the network.
sigmoid' :: Matrix R -> Matrix R
sigmoid' = cmap (\x -> x * (1 - x))

-- The ReLU activation function. This is actually the leaky ReLU function to avoid the "dying ReLU" problem.
relu :: Matrix R -> Matrix R
relu = cmap (\x -> if x < 0 then 0.01 * x else x)

-- The derivative for leaky ReLU.
relu' :: Matrix R -> Matrix R
relu' = cmap (\x -> if x < 0 then 0.01 else 1)

-- The softmax activation function
softmax :: Matrix R -> Matrix R
softmax mat = scale (1/ s) exps
  where 
    s = sumElements exps
    exps = cmap exp mat

softmax' :: Matrix R -> Matrix R
softmax' mat = diagS - outerS
  where 
    s = flatten $ softmax mat
    diagS = diag s
    outerS = outer s s

-- The mean squared error function.
mse :: [Matrix R] -> [Matrix R] -> R
mse output target = (1 / (2 * n)) * totalLoss
 where
  totalLoss :: R
  totalLoss = sum [sumElements (o - t) ** 2 | (o, t) <- zip output target]
  n = fromIntegral $ fst (size $ head target) :: R

-- Converts a matrix into a list of column vectors from its rows
matrixToRows :: Matrix R -> [Matrix R]
matrixToRows x = map (tr . asRow) (toRows x)

-- Helper function for gradient clipping.
clipGradient :: R -> R -> R -> R
clipGradient minVal maxVal x
  | x < minVal = minVal
  | x > maxVal = maxVal
  | otherwise = x

-- Helps avoid gradient explosion, where the gradients are so large that the weight parameters get insanely high values.
clipGradients :: [(Matrix R, Matrix R)] -> [(Matrix R, Matrix R)]
clipGradients =
  map
    ( bimap
        (cmap (clipGradient (-1) 1))
        (cmap (clipGradient (-1) 1))
    )

xorInput :: Matrix R
xorInput =
  (4 >< 2)
    [ 0
    , 0
    , 1
    , 0
    , 0
    , 1
    , 1
    , 1
    ] ::
    Matrix R

xorOutput :: Matrix R
xorOutput =
  (4 >< 1)
    [ 0
    , 1
    , 1
    , 0
    ]
