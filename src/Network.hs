module Network
    ( Network
    , Layer
    , sigmoid
    , sigmoid'
    , initialize
    , fit
    , weights
    , biases
    --, train
    , predict
    , updateParams
    , forwardProp
    , backProp
    , eval
    --, trainXOR
    , printNet
    ) where

import Numeric.LinearAlgebra
import qualified Numeric.LinearAlgebra as LA
import Util (enumerate, sigmoid, sigmoid', mse)
import System.Random (randomRIO)
import Control.Monad (replicateM)

data Layer = Layer
    { weights :: Matrix R
    , biases :: Matrix R
    , sz:: Int
    , activation :: R -> R
    , activation' :: R -> R
    }

type Network = [Layer]

activate :: Layer -> Matrix R -> Matrix R
activate l x = cmap (activation l) (weights l LA.<> x + biases l)

xavierInit :: Int -> Int -> IO (Matrix R)
xavierInit n m = do
    let limit = sqrt (2.0 / fromIntegral n)
    values <- replicateM (n * m) (randomRIO (-limit, limit))
    return $ (n><m) values

initWeights :: Int -> Int -> Matrix R
initWeights n m = (n><m) $ replicate (n * m) 0.0

initBiases :: Int -> IO (Matrix R)
initBiases n = do
    values <- replicateM n (randomRIO (-0.1, 0.1))
    return $ (n><1) values

initialize :: [Int] -> [R -> R] -> [R -> R] -> IO Network
initialize s f d = mapM (\(x, idx) -> do
    let w = initWeights 0 0
    b <- initBiases 0
    return Layer {weights = w, biases = b, sz = x, activation = f!!idx, activation' = d!!idx}) $ enumerate s


fit :: Matrix R -> Network -> IO Network
fit x n = mapM (\(layer, idx) -> do
    w <- xavierInit (sz layer) (len idx)
    b <- initBiases $ sz layer
    return Layer {weights = w, biases = b, sz = sz layer, activation = activation layer, activation' = activation' layer}) (enumerate n)
    where
        len idx = if idx == 0 then fst (size x) else sz $ n!!(idx - 1)

forwardProp :: Matrix R -> Network -> [Matrix R]
forwardProp = scanl (flip activate)

-- output is from previous layer (AKA the input to this layer)
calculateDelta :: (Layer, Matrix R, Layer) -> Matrix R -> Matrix R
calculateDelta (l, output, nextL) deltaNext = (tr (weights nextL) LA.<> deltaNext) * cmap (activation' l) output

-- `outputs` comes right from forwardProp
-- Return a list of tuples with weights and biases for every layer
backProp :: [Matrix R] -> Matrix R -> Network -> [(Matrix R, Matrix R)]
backProp outputs target n = zip dW dB
    where
        outputError = (last outputs - target) * cmap sigmoid' (last outputs)

        reversedLayers = tail $ reverse n
        reversedOutputs = tail $ reverse outputs
        reversedNextLayers = reverse $ tail n

        layersWithOutputs = zip3 reversedLayers reversedOutputs reversedNextLayers
        deltas = reverse $ scanl (flip calculateDelta) outputError layersWithOutputs

        dW = [delta LA.<> tr out | (delta, out) <- zip deltas (init outputs)]
        dB = deltas

-- Network -> [(weight matrix for the l:th layer, bias vector (n x 1 matrix) for the l:th layer
updateParams :: R -> [(Matrix R, Matrix R)] -> Network -> Network
updateParams lr params n = zipWith (curry (\(l, (dW, dB)) -> Layer {weights=weights l - scale lr dW, biases=biases l - scale lr dB, sz=sz l, activation=activation l, activation' = activation' l})) n params

--trainStep :: Matrix R -> Matrix R -> R -> Network -> Network
--trainStep x y lr n = updateParams lr totalParams n
--    where
--        inputs = matrixToRows x
--        outputs = matrixToRows y

--        gradients = [backProp (forwardProp input n) output n | (input, output) <- zip inputs outputs]
--        totalParams = clipGradients $ map f $ transpose gradients
--        f gs = let len = fromIntegral (length gs) in foldl1 (\(accW, accB) (w, b) -> (accW + w / len, accB + b / len)) gs

-- net -> x -> y -> learning rate -> epochs
predict :: Matrix R -> Network -> Matrix R
predict x n = last $ forwardProp x n


printNet :: Network -> IO ()
printNet = mapM_ (\l -> do
    putStrLn $ "Weights: " ++ show (weights l)
    putStrLn $ "Biases: " ++ show (biases l))

eval :: Network -> [Matrix R] -> [Matrix R] -> R
eval n x = mse output
    where
        output = map (`predict` n) x
        --output = map (\v -> predict v n) x
