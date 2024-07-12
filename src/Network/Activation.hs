module Network.Activation (Activation, sigmoid, relu, softmax, derivative, function) where

import Numeric.LinearAlgebra

data Activation = Activation
  { function :: Matrix R -> Matrix R
  , derivative :: Matrix R -> Matrix R
  }

sigmoid :: Activation
sigmoid = Activation{function = fSigmoid, derivative = fSigmoid'}

relu :: Activation
relu = Activation{function = fRelu, derivative = fRelu'}

softmax :: Activation
softmax = Activation{function = fSoftmax, derivative = fSoftmax'}

-- The sigmoid function.
fSigmoid :: Matrix R -> Matrix R
fSigmoid = cmap (\x -> 1 / (1 + exp (-x)))

-- The sigmoid derivative is defined as sigmoid(x) * (1 - sigmoid(x)).
-- Here we just assume the argument to the function has already been "sigmoided" - as it is when calling this function in the network.
fSigmoid' :: Matrix R -> Matrix R
fSigmoid' = cmap (\x -> x * (1 - x))

-- The ReLU activation function. This is actually the leaky ReLU function to avoid the "dying ReLU" problem.
fRelu :: Matrix R -> Matrix R
fRelu = cmap (\x -> if x < 0 then 0.01 * x else x)

-- The derivative for leaky ReLU.
fRelu' :: Matrix R -> Matrix R
fRelu' = cmap (\x -> if x < 0 then 0.01 else 1)

-- The softmax activation function
fSoftmax :: Matrix R -> Matrix R
fSoftmax mat = scale (1 / s) exps
  where
    s = sumElements exps
    exps = cmap exp mat

fSoftmax' :: Matrix R -> Matrix R
fSoftmax' mat = diagS - outerS
  where
    s = flatten $ fSoftmax mat
    diagS = diag s
    outerS = outer s s
