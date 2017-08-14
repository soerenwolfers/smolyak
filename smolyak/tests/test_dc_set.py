'''
Test smolyak.sparse.dc_set
'''
import unittest
from smolyak.indices import MultiIndex, DCSet
class TestDCSet(unittest.TestCase):

    def test1(self):
        A = DCSet()
        A.add(MultiIndex([0]))

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
