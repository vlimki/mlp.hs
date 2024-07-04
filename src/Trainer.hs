module Trainer 
    ( Trainer
    , train
    , learningRate
    , epochs
    , bgdTrainer
    ) where

import Numeric.LinearAlgebra
import Network
import Util
import Data.List (transpose)

class Trainer t where
    train :: t -> Network -> Matrix R -> Matrix R -> Network

-- Batch Gradient Descent trainer
data BGDTrainer = BGDTrainer
    { learningRate :: R
    , epochs :: Int
    }

instance Trainer BGDTrainer where
    --train :: BGDTrainer -> Network -> Matrix R -> Matrix R -> Network
    train (BGDTrainer _ 0) n _ _  = n
    train (BGDTrainer lr e) n x y = train (BGDTrainer lr (e - 1)) (trainStep lr n) x y
        where 
            trainStep :: R -> Network -> Network
            trainStep lr' = updateParams lr' totalParams
            inputs = matrixToRows x
            outputs = matrixToRows y
            gradients = [backProp (forwardProp input n) output n | (input, output) <- zip inputs outputs]
            totalParams = clipGradients $ map f $ transpose gradients
            f gs = let len = fromIntegral (length gs) in foldl1 (\(accW, accB) (w, b) -> (accW + w / len, accB + b / len)) gs

bgdTrainer :: R -> Int -> BGDTrainer
bgdTrainer lr e = BGDTrainer {learningRate=lr, epochs=e}
