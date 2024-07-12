module Util (
  enumerate,
  mse,
  matrixToRows,
  xorInput,
  xorOutput,
  clipGradients,
)
where

import Data.Bifunctor
import Numeric.LinearAlgebra

-- For iterating arrays with indices
enumerate :: [a] -> [(a, Int)]
enumerate a = zip a [0 .. length a]

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
