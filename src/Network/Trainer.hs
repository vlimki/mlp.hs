module Network.Trainer
  ( Trainer
  , train
  , learningRate
  , epochs
  , bgdTrainer
  ) where

import Data.List (transpose)
import Network.Network
import Numeric.LinearAlgebra
import Util

-- A general trainer typeclass. Useful for when I'll eventually add other training algorithms like SGD.
class Trainer t where
  train :: t -> Network -> Matrix R -> Matrix R -> Network

-- Batch Gradient Descent trainer
data BGDTrainer = BGDTrainer
  { learningRate :: R
  , epochs :: Int
  }

instance Trainer BGDTrainer where
  train (BGDTrainer _ 0) n _ _ = n
  train (BGDTrainer lr e) n x y = train (BGDTrainer lr (e - 1)) (trainStep lr n) x y
    where
      trainStep :: R -> Network -> Network
      trainStep lr' = updateParams lr' totalParams

      -- Convert the input and output matrices to lists of column vectors.
      inputs = matrixToRows x
      outputs = matrixToRows y

      -- Loops through the dataset in (input, output) pairs and calculates the gradients for each element
      -- `gradients` has the type [[(Matrix R, Matrix R)]], where the outer list represents every training example and the inner list represents every layer.
      gradients = [backProp (forwardProp input n) output n | (input, output) <- zip inputs outputs]

      -- Here we get the average of all the gradients for every layer from every training example.
      -- `gradients` is in the form [[(a, b), (a, b)]], where a is the weight matrix and b is the bias matrix.
      -- The `transpose` function we called above turns it from that form to the form [[(a, a), (b, b)]]. This way it's easier to calculate to sum them both up.
      f gs = let len = fromIntegral (length gs) in foldl1 (\(accW, accB) (w, b) -> (accW + w / len, accB + b / len)) gs

      -- Calculate the average gradients and apply gradient clipping.
      totalParams = clipGradients $ map f $ transpose gradients

-- Helper function to initialize a BGD trainer.
bgdTrainer :: R -> Int -> BGDTrainer
bgdTrainer lr e = BGDTrainer{learningRate = lr, epochs = e}
