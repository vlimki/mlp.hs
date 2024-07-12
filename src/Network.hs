module Network
  ( Network,
    Layer,
    sigmoid,
    sigmoid',
    initialize,
    fit,
    weights,
    biases,
    predict,
    updateParams,
    forwardProp,
    backProp,
    eval,
    printNet,
  )
where

import Control.Monad (replicateM)
import Numeric.LinearAlgebra
import qualified Numeric.LinearAlgebra as LA
import System.Random (randomRIO)
import Util (enumerate, mse, sigmoid, sigmoid')

-- The layer type. The parameters in the network are stored on a layer basis.
-- weights     = weight matrix
-- biases      = bias matrix (it's an (n x 1) matrix, so it could totally well be a vector, but it's easier to just put a matrix there
-- sz          = number of neurons
-- activation  = the activation function
-- activation' = the derivative of the activation function
data Layer = Layer
  { weights :: Matrix R,
    biases :: Matrix R,
    sz :: Int,
    activation :: Matrix R -> Matrix R,
    activation' :: Matrix R -> Matrix R
  }

-- A network is just a list of layers.
type Network = [Layer]

-- The `activate` function is basically just g(z), where:
-- g = the activation function of the layer
-- z = w.
-- The activation function is applied on an element basis to the output vector.
activate :: Layer -> Matrix R -> Matrix R
activate l x = activation l (weights l LA.<> x + biases l)

-- The xavier weight initialization method. It works well with the sigmoid activation function.
-- n = the number of neurons
-- m = the number of features in the input vector
xavierInit :: Int -> Int -> IO (Matrix R)
xavierInit n m = do
  let limit = sqrt (2.0 / fromIntegral n)
  values <- replicateM (n * m) (randomRIO (-limit, limit))

  -- Build a matrix from the list of values that `replicateM` outputs
  return $ (n >< m) values

-- Just initializing every weight to a constant 0. This is a bad practice but it's here anyway for testing purposes.
-- n = the number of neurons
-- m = the number of features in the input vector
initWeights :: Int -> Int -> Matrix R
initWeights n m = (n >< m) $ replicate (n * m) 0.0

-- Initialize every bias to a random value from the range [-0.1, 0.1].
initBiases :: Int -> IO (Matrix R)
initBiases n = do
  values <- replicateM n (randomRIO (-0.1, 0.1))
  return $ (n >< 1) values

-- Initialize the network. Note that the dimensions of the weight and bias matrices aren't calculated here yet.
-- This only initializes the layer structures with their intended numbers of neurons and activation functions.
initialize :: [Int] -> [Matrix R -> Matrix R] -> [Matrix R -> Matrix R] -> IO Network
initialize s f d =
  mapM
    ( \(x, idx) -> do
        let w = initWeights 0 0
        b <- initBiases 0
        return Layer {weights = w, biases = b, sz = x, activation = f !! idx, activation' = d !! idx}
    )
    $ enumerate s

-- This is where the dimensions of the weight and bias matrices get initialized.
fit :: Matrix R -> Network -> IO Network
fit x n =
  mapM
    ( \(layer, idx) -> do
        w <- xavierInit (sz layer) (len idx)
        b <- initBiases $ sz layer
        return layer {weights = w, biases = b, sz = sz layer}
    )
    (enumerate n)
  where
    len idx = if idx == 0 then fst (size x) else sz $ n !! (idx - 1)

-- Surprisingly short code for a forward propagation function - `scanl` iterates through the layers and activates them with the output of the previous layer.
-- It's equivalent to `scanl (flip activate) input network`, where `input` is the initial value that `activate` gets called on.
forwardProp :: Matrix R -> Network -> [Matrix R]
forwardProp = scanl (flip activate)

-- (l, input, nextL) = (the current layer, the output of the from the previous layer, the next layer
calculateDelta :: (Layer, Matrix R, Layer) -> Matrix R -> Matrix R
calculateDelta (l, input, nextL) deltaNext = (tr (weights nextL) LA.<> deltaNext) * activation' l input

-- The backpropagation algorithm returns a list of tuples with weights and biases for every layer.
-- outputs = the output from the forwardProp function - so a list of output vectors
-- target = the target output
backProp :: [Matrix R] -> Matrix R -> Network -> [(Matrix R, Matrix R)]
backProp outputs target n = zip dW dB
  where
    outputError = (last outputs - target) * sigmoid' (last outputs)

    reversedLayers = tail $ reverse n
    reversedOutputs = tail $ reverse outputs
    reversedNextLayers = reverse $ tail n

    layersWithOutputs = zip3 reversedLayers reversedOutputs reversedNextLayers
    deltas = reverse $ scanl (flip calculateDelta) outputError layersWithOutputs

    dW = [delta LA.<> tr out | (delta, out) <- zip deltas (init outputs)]
    dB = deltas

-- This maps through the layers structures and replaces them with new ones with slightly adjusted parameters.
-- params = [(weight matrix for the l:th layer, bias vector for the l:th layer)]
updateParams :: R -> [(Matrix R, Matrix R)] -> Network -> Network
updateParams lr params n = zipWith (curry (\(l, (dW, dB)) -> Layer {weights = weights l - scale lr dW, biases = biases l - scale lr dB, sz = sz l, activation = activation l, activation' = activation' l})) n params

-- This just takes the last element of the output of the `forwardProp` function - so the output of the network.
predict :: Matrix R -> Network -> Matrix R
predict x n = last $ forwardProp x n

-- Debug printing
printNet :: Network -> IO ()
printNet =
  mapM_
    ( \l -> do
        putStrLn $ "Weights: " ++ show (weights l)
        putStrLn $ "Biases: " ++ show (biases l)
    )

-- Test the accuracy of the model with the mean squared error loss function.
eval :: Network -> [Matrix R] -> [Matrix R] -> R
eval n x = mse output
  where
    output = map (`predict` n) x

-- output = map (\v -> predict v n) x
