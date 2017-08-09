'''
Vector valued function
'''
class VFunction(object):
    '''
    Vector-valued function that supports operations of vector space of functions with same codomain
    
    :param function: Function that is to be equipped with vector space functionality
    '''
    def __init__(self, function):
        self.functions = [function]
        self.multipliers = [1]
        
    def __call__(self, X):
        y = list()
        for i, f in enumerate(self.functions):
            y.append(self.multipliers[i] * f(X))
        return sum(y)
        
    def __add__(self, other):
        new = self.copy()
        new.functions += other.functions
        new.multipliers += other.multipliers
        return new
    
    def __radd__(self, other):
        if other == 0:
            T = self.copy()
            return T
        else: 
            T = self.__add__(other)
            return T
        
    # def __iadd__(self,other):
    #    return self.__add__(other)
    
    def __rmul__(self, other):
        new = self.copy()
        new.multipliers = [m * other for m in new.multipliers]
        return new
    
    def copy(self):
        import copy
        new = VFunction(None)
        new.functions = copy.deepcopy(self.functions)
        new.multipliers = copy.deepcopy(self.multipliers)
        return new