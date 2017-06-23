'''
Config class stores global and local options associated with each run. 

E.g.
config=Config()
config['reltol']=10e-2
do_stuff(config)
A=GreatObject(config) #options added by A will be visible everywhere 
A=GreatObject(config.fork('GreatThings')) #A gets its own virtual environment, 
                                        initialized with and stored within config
A=GreatObject(config.spawn('GreaterThings')) #A gets its own, empty, virtual 
                                            environment, stored within config
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
    def __init__(self,rules=None):
        self.__rules=rules or []
        self.dict={}

    def __setitem__(self,name,value=None):
        for rule in self.__rules:
            if len(inspect.getargspec(rule).args)==2:
                (name,value)=rule(name,value)
            else:
                (name,value)=rule(name,value,self.dict)
        self.dict[name]=value
            
    def __contains__(self,name):
        return self.dict.__contains__(name)
    
    def __getitem__(self,name):
        return self.dict[name]
    
    def __delitem__(self,name):
        pass
    
    def __iter__(self):
        return self.dict.__iter__()
    
    def __repr__(self, *args, **kwargs):
        return self.dict.__repr__()
    
    def __str__(self):
        return self.dict.__str__()
    
    def enforce_rule(self,rule):
        self.enforce_rules([rule])
            
    def enforce_rules(self,rules=[]):
        self.__rules+=rules
        dict_old = self.dict
        self.dict={}
        for option in dict_old:
            self[option]=dict_old[option]
            
    def forget_rule(self,rule):
        self.__rules=[_rule for _rule in self.__rules if _rule is not rule]
        
    def forget_rules(self,rules):
        for rule in rules:
            self.forget_rule(rule)

    def set_defaults(self,defaults):
        '''
        Set options but only if they haven't already been defined
        
        :param defaults: Options and values to be set
        :type defaults: behaving like dict or Config
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
