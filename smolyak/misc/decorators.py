import cProfile
import pstats
import math

def log_instance_method(function):
    '''
    Decorator that logs function calls in their self.log
    '''
    def wrapper(self,*args,**kwargs):
        self.log.log(group=function.__name__,message='Enter')
        function(self,*args,**kwargs)
        self.log.log(group=function.__name__,message='Exit')
    return wrapper

def add_profile(function):
    '''
    Decorator that adds a profile object
    '''
    def wrapper(*args,**kwargs):
        pr=cProfile.Profile()
        pr.enable()
        output = function(*args,**kwargs)
        pr.disable()
        return pr,output
    return wrapper

def print_stats(function):
    '''
    Decorator that prints stats at each call of the function
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

if __name__=='__main__':
    @print_stats
    def test(a):
        return math.factorial(a)
    print(test(3))
    print(test(40))
        