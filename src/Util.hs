module Util
    ( enumerate
    , sigmoid
    , sigmoid'
    , mse
    , matrixToRows
    , xorInput
    , xorOutput
    )
where

import Numeric.LinearAlgebra

enumerate :: [a] -> [(a, Int)]
enumerate a = zip a [0..length a]

sigmoid :: R -> R
sigmoid x = 1 / (1 + exp (-x))

-- Assuming the argument has alreadh been "sigmoided"
sigmoid' :: R -> R
sigmoid' x = sigmoid x * (1 - sigmoid x)


-- vector of predictions -> y vector
mse :: Vector R -> Vector R -> R
mse yHatVector yVector = (1/(2*n)) * totalLoss
    where
        totalLoss = sum $ toList $ (yHatVector - yVector) ** 2
        n = fromIntegral (size yHatVector) :: R

-- Converts a matrix into column vectors from its rows
matrixToRows :: Matrix R -> [Matrix R]
matrixToRows x = map tr $ map asRow $ toRows x

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
