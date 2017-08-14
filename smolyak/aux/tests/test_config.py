import unittest
from smolyak.aux.config import Config, lower, to_bool, to_string, to_float,\
    final

class TestConfig(unittest.TestCase):
    def test_1(self):
        a=1
        def to_float(option,value):
            if type(value) is int:
                value=float(value)
            return (option,value)
            
        opts=Config()
        opts.set_defaults({'testing':'False','truth':123,'integer':123,a:1288})
        rules=[lower,to_float]
        print(str(opts))
        print(opts)
        opts.enforce_rule(to_bool)
        print(opts)
        opts2=Config(rules=rules)
        opts2.set_defaults(opts)
        print(opts2)
        
        opts=Config()
        opts.set_defaults({'testing':'True','test':125})
        opts['great']='Yo'
        fork=opts.fork('fork')
        print(opts._dict)
        print(opts['great'])
        print(fork['great'])
        print(opts['fork']['great'])
        
    def test_rules(self):
        rules=[to_string,to_float]
        opts=Config(rules=rules)
        opts[12]=12
        opts['False']='True'
        print(opts)
        
        opts.enforce_rule(final)
        def failure():
            opts['False']=12
        self.assertRaises(ValueError,failure)
        opts.enforce_rules({final,to_bool})
        opts.enforce_rules()
        print(opts._Config__rules)
        opts.forget_rule(final)
        
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test1']
    unittest.main()
    