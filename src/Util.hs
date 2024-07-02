module Util
    ( enumerate
    , sigmoid
    , sigmoid'
    , mse
    )
where

import Numeric.LinearAlgebra

enumerate :: [a] -> [(a, Int)]
enumerate a = zip a [0..length a]

sigmoid :: R -> R
sigmoid x = 1 / (1 + exp (-x))

-- Assuming the argument has alreadh been "sigmoided"
sigmoid' :: R -> R
sigmoid' x = x * (1 - x)


-- vector of predictions -> y vector
mse :: Vector R -> Vector R -> R
mse yHatVector yVector = (1/(2*n)) * totalLoss
    where 
        totalLoss = sum $ toList $ (yHatVector - yVector) ** 2
        n = fromIntegral (size yHatVector) :: R
