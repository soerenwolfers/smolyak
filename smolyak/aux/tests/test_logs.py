'''
Test smolyak.aux.logs
'''
import unittest
from smolyak.aux.logs import Log

class TestLogs(unittest.TestCase):

    def test_Log(self):
        log=Log()
        log.log(message='Wait')
        log.log(message='Cleanup')
        log.log(group='3',message='whatdup')
        log.print_entries(require_message='up')
        print(log)
    

if __name__ == "__main__":
    unittest.main()
