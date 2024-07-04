module Lib (fit, train, bgdTrainer, initialize, predict, trainXOR) where

import Network
import Trainer
import Util

trainXOR :: IO Network
trainXOR = do
    n1 <- initialize [4, 1] [relu, sigmoid] [relu', sigmoid']
    n2 <- fit (head $ matrixToRows xorInput) n1

    putStrLn "Initial weights:"
    printNet n2

    let t = bgdTrainer 0.1 10000
    let n3 = train t n2 xorInput xorOutput

    putStrLn "Final weights:"
    printNet n3

    putStrLn "Predictions:"
    mapM_ (\input -> putStrLn $ "Input: " ++ show input ++ ", Output: " ++ show (predict input n3)) (matrixToRows xorInput)

    let loss = eval n3 (matrixToRows xorInput) (matrixToRows xorOutput)
    putStrLn $ "Loss: " ++ show loss

    return n3
