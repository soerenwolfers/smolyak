'''
Test smolyak.aux.decorators
'''
import unittest
from smolyak.aux.decorators import print_profile
import math

class TestDecorators(unittest.TestCase):

    def test_print_profile(self):
        @print_profile
        def test(a):
            return math.factorial(a)
        test(3)
        test(20000)

if __name__ == "__main__":
    unittest.main()

