'''
Config class stores global and local options associated with each run. 

config=Config()
config['reltol']=10e-2
foo(config) #options added by foo will be visible everywhere 
foo(config.sub('GreaterThings')) #foo gets its own empty Config instance
                                 # which is stored within config
foo(config.fork('GreatThings')) #foo gets its own Config instance
                                # which is initialized with the values of config
                                # and stored within config

'''
from copy import deepcopy
import inspect

'''
Rules that can be passed to Config()
'''
   
def lower(option,value):
    '''
    Enforces lower case options and option values where appropriate
    '''
    if type(option) is str:
        option=option.lower()
    if type(value) is str:
        value=value.lower()
    return (option,value)

def to_string(option,value):
    '''
    Converts any values to strings when appropriate
    '''
    try:
        value=value.__str__()
    except AttributeError:
        pass
    return (option,value)
    
def to_float(option,value):
    '''
    Converts string values to floats when appropriate
    '''
    if type(value) is str:
        try:
            value=float(value)
        except ValueError:
            pass
    return (option,value)

def to_bool(option,value):
    '''
    Converts string values to booleans when appropriate
    '''
    if type(value) is str:
        if value.lower() == 'true':
            value=True
        elif value.lower() == 'false':
            value=False
    return (option,value)

def final(option,value,config):
    if option in config:
        raise ValueError('Cannot change option value')
    else:
        return (option,value)
    
class Config(object):
    def __init__(self,config=None,rules=None):
        '''
        :param config: Old configuration
        :type config: behaving like dict or Config
        '''
        self._rules = rules or []
        self._dict = config or {}
        self.enforce_rules()

    def __setitem__(self,name,value=None):
        for rule in self._rules:
            if len(inspect.getargspec(rule).args)==2:
                (name,value)=rule(name,value)
            else:
                (name,value)=rule(name,value,self._dict)
        self._dict[name]=value
            
    def __contains__(self,name):
        return self._dict.__contains__(name)
    
    def __getitem__(self,name):
        return self._dict[name]
    
    def __delitem__(self,name):
        pass
    
    def __iter__(self):
        return self._dict.__iter__()
    
    def __repr__(self, *args, **kwargs):
        return self._dict.__repr__()
    
    def __str__(self):
        return self._dict.__str__()
    
    def enforce_rule(self,rule):
        self.enforce_rules([rule])
            
    def enforce_rules(self,rules=[]):
        self._rules+=rules
        dict_old = self._dict
        self._dict={}
        for option in dict_old:
            self[option]=dict_old[option]
            
    def forget_rule(self,rule):
        self._rules=[_rule for _rule in self._rules if _rule is not rule]
        
    def forget_rules(self,rules):
        for rule in rules:
            self.forget_rule(rule)

    def set_defaults(self,defaults):
        '''
        Set options but only if they haven't already been defined
        
        :param defaults: Options and values to be set
        :type defaults: behaving like _dict or Config
        '''
        self._set_defaults(self,defaults)
    
    def _set_defaults(self,_dict,defaults):
        for option in defaults:
            if not option in _dict:
                _dict[option]=defaults[option]

    def fork(self,name):
        '''
        Create fork and store it in current instance
        '''
        fork=deepcopy(self)
        self[name]=fork
        return fork
    
    def sub_config(self,name):
        '''
        Return empty Config object and store this object in current Config
        '''
        sub = Config()
        self[name]=sub
        return sub
        
