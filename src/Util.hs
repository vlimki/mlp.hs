{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant bracket" #-}
module Util
    ( enumerate
    , sigmoid
    , sigmoid'
    , mse
    , matrixToRows
    , xorInput
    , xorOutput
    , relu
    , relu'
    , clipGradients
    )
where

import Numeric.LinearAlgebra
import Data.Bifunctor

enumerate :: [a] -> [(a, Int)]
enumerate a = zip a [0..length a]

sigmoid :: R -> R
sigmoid x = 1 / (1 + exp (-x))

-- Assuming the argument has alreadh been "sigmoided"
sigmoid' :: R -> R
sigmoid' x = x * (1 - x)

relu :: R -> R
relu x
    | x < 0 = 0.01 * x
    | otherwise = x

relu' :: R -> R
relu' x
    | x < 0 = 0.01
    | otherwise = 1

-- vector of predictions -> y vector
mse :: [Matrix R] -> [Matrix R] -> R
mse output target = (1/(2*n)) * totalLoss
    where
        totalLoss :: R
        totalLoss = sum [(sumElements (o - t) ** 2) | (o, t) <- zip output target]
        n = fromIntegral $ fst (size $ head target) :: R

-- Converts a matrix into column vectors from its rows
matrixToRows :: Matrix R -> [Matrix R]
matrixToRows x = map (tr . asRow) (toRows x)

clipGradient :: R -> R -> R -> R
clipGradient minVal maxVal x
  | x < minVal = minVal
  | x > maxVal = maxVal
  | otherwise  = x

clipGradients :: [(Matrix R, Matrix R)] -> [(Matrix R, Matrix R)]
clipGradients = map (bimap
  (cmap (clipGradient (- 5) 5)) (cmap (clipGradient (- 5) 5)))

xorInput :: Matrix R
xorInput = (4><2)
    [ 0, 0
    , 1, 0
    , 0, 1
    , 1, 1] :: Matrix R

xorOutput :: Matrix R
xorOutput = (4><1)
    [ 0
    , 1
    , 1
    , 0 ]
