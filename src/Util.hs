module Util
    ( enumerate
    )
where

enumerate :: [a] -> [(a, Int)]
enumerate a = zip a [0..length a]
