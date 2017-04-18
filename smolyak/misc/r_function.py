'''
Real valued function
'''
class RFunction(dict):
    '''
    Real-valued function that supports operations of vector space of functions
    
    Instances are called and modified using __*etitem__()
    '''
    def __init__(self, init_dict=None):
        '''
        :param init_dict: Initial state of function
        :type init_dict: Dictionary whos values support addition and scalar multiplication 
        '''
        dict.__init__(self)
        if init_dict:
            for key in init_dict:
                self[key] = init_dict[key]
                
    def expand_domain(self, X):
        '''
        Expand domain
        
        :param X: New elements of domain
        :type X: Iterable
        '''
        for x in X:
            self[x] = None
        
    def __add__(self, other):
        '''
        Vector space operation: Add two real-valued functions
        '''
        F = RFunction()
        for key in self.keys():
            F[key] = self[key]
        for key in other.keys():
            if key in F.keys():
                F[key] += other[key]
            else:
                F[key] = other[key]
        return F
    
    def __radd__(self, other):
        '''
        When iterables of functions are added, the first function is added to 0
        using __radd__
        '''
        if other == 0:
            return self
        else: 
            return self.__add__(other)
        
    def __iadd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        '''
        Vector space operation: Multiply real-valued function with real
        '''
        F = RFunction()
        for key in self.keys():
            F[key] = other * self[key]
        return F
