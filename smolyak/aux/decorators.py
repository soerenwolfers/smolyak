import cProfile
import pstats
from _io import StringIO


def log_calls(function):
    '''
    Decorator that logs function calls in their self.log
    '''
    def wrapper(self,*args,**kwargs):  
        self.log.log(group=function.__name__,message='Enter') 
        function(self,*args,**kwargs)
        self.log.log(group=function.__name__,message='Exit') 
    return wrapper

def add_runtime(function):
    '''
    Decorator that adds a runtime profile object to the output
    '''
    def wrapper(*args,**kwargs):  
        pr=cProfile.Profile()
        pr.enable()
        output = function(*args,**kwargs)
        pr.disable()
        return pr,output
    return wrapper

def print_memory(function):
    '''
    Decorator that prints memory information at each call of the function
    '''
    import memory_profiler
    def wrapper(*args,**kwargs):
        m = StringIO()
        temp_func = memory_profiler.profile(func = function,stream=m,precision=4)
        output = temp_func(*args,**kwargs)
        print(m.getvalue())
        m.close()
        return output
    return wrapper
    
def print_profile(function):
    '''
    Decorator that prints memory and runtime information at each call of the function
    '''
    import memory_profiler
    def wrapper(*args,**kwargs):
        m=StringIO()
        pr=cProfile.Profile()
        pr.enable()
        temp_func = memory_profiler.profile(func=function,stream=m,precision=4)
        output = temp_func(*args,**kwargs)
        print(m.getvalue())
        pr.disable()
        ps = pstats.Stats(pr)
        ps.sort_stats('cumulative').print_stats('(?!.*memory_profiler.*)(^.*$)',20)
        m.close()
        return output
    return wrapper

def print_runtime(function):
    '''
    Decorator that prints running time information at each call of the function
    '''
    def wrapper(*args,**kwargs):
        pr=cProfile.Profile()
        pr.enable()
        output = function(*args,**kwargs)
        pr.disable()
        ps = pstats.Stats(pr)
        ps.sort_stats('cumulative').print_stats(20)
        return output
    return wrapper
