'''
Test smolyak.experiments
'''
import unittest
from smolyak.experiments import conduct,load
import numpy as np
class TestTestSeries(unittest.TestCase):
    def test(self):
        def np_rand(opts):
            X=np.random.rand(opts['n'],opts['n'])
            return X
        tests=[{'n':int(10**(i/2.0))} for i in range(9)]
        path=conduct(func=np_rand,tests=tests,overwrite=True,memory_profile=True)
        __,__=load(path)

if __name__ == "__main__":
    unittest.main()
