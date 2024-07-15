module Main (main) where

import Lib

main :: IO ()
main = do 
  _ <- trainMNIST
  return ()
