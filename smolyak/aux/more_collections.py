from __future__ import absolute_import
import collections

def unique(seq):
    '''
    https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order
    '''
    has = []
    return [x for x in seq if not (x in has or has.append(x))]

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

class OrderedSet(collections.MutableSet):
    '''
    http://code.activestate.com/recipes/576694/ under MIT license
    '''
    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)  # @ReservedAssignment
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)
    
class DefaultDict(dict):
    '''
    Dictionary that returns default value on unknown keys
    '''
    def __init__(self, default):
        '''
        :param default: Default values for unknown keys
        :type default: Function
        '''
        self.default = default
        dict.__init__(self)

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            result = self[key] = self.default(key)
            return result