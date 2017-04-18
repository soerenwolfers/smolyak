'''
Test smolyak.misc.v_function
'''
import unittest
from smolyak.misc.v_function import VFunction

class TestVFunction(unittest.TestCase):

    def test_lambdas(self):
        A = lambda x: x ** 2
        B = lambda x:-x ** 2 + 1
        F = VFunction(A)
        G = VFunction(B)
        H = F + G
        self.assertEqual(H(5.), 1.)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test1']
    unittest.main()
