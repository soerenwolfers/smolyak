'''
Test smolyak.misc.snooze
'''
import unittest
import timeit
from smolyak.misc.snooze import snooze

class TestSnooze(unittest.TestCase):

    def test1(self):
        tic = timeit.default_timer()
        snooze(100)
        toc = timeit.default_timer() - tic
        tic = timeit.default_timer()
        snooze(200)
        toc2 = timeit.default_timer() - tic
        self.assertAlmostEqual(toc2 / toc, 2, delta=0.1)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test1']
    unittest.main()
