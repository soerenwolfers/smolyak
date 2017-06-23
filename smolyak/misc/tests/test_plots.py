'''
Test smolyak.misc.plots
'''
import unittest
from smolyak.misc.plots import plot_convergence

class TestPlots(unittest.TestCase):

    def test_plotConvergence(self):
        order = plot_convergence([1, 2, 4, 8, 16, 32], [2, 0.3, 0.2, 0.135, 0.0525, 0.03125],
                             expect_order='fit', expect_limit=0, p=10)
        self.assertAlmostEqual(order, -1., delta=0.1)

if __name__ == "__main__":
    unittest.main()
