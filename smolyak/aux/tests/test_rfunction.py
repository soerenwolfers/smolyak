'''
Test smolyak.aux.r_function
'''
import unittest
from smolyak.aux.more_collections import RFunction

class TestRFunction(unittest.TestCase):

    def test_real_arithmetic(self):
        F = RFunction()
        F[1] = 2
        F[2] = 3
        G = 3 * F
        H = G + F
        self.assertEqual(F[1], 2)
        self.assertEqual(F[2], 3)
        self.assertEqual(G[1], 6)
        self.assertEqual(G[2], 9)
        self.assertEqual(H[1], 8)
        self.assertEqual(H[2], 12)
        
    def test_domains(self):
        F = RFunction()
        F[(1, 2)] = 3
        G = -1 * F
        self.assertEqual(G[(1, 2)], -3)

if __name__ == "__main__":
    unittest.main()
