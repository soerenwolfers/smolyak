'''
Test smolyak.aux.grid_evaluation
'''
import unittest
import numpy as np
from smolyak.aux.np_tools import grid_evaluation

class TestGridEvaluation(unittest.TestCase):

    def test_shape(self):
        grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
        def function(X):
            return X[:, 0] * X[:, 1]
        self.assertEqual(grid_evaluation(grid_x, grid_y, function).shape, grid_x.shape)
    
    def test_values(self):
        grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
        def function1(X):
            return X[:, 0]
        self.assertEqual(grid_evaluation(grid_x, grid_y, function1)[10, 0], grid_x[10, 0])
        def function2(X):
            return X[:, 1]
        self.assertEqual(grid_evaluation(grid_x, grid_y, function2)[5, 0], grid_y[5, 0])
      
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test1']
    unittest.main()
