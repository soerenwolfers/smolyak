'''
Sparse multi-indices
'''
import itertools
import numpy as np

class SparseIndex(object):
    '''
    Sparse multi-index representation.
    
    Dimension of multi-index is implicitly infinity, since non-set entries are assumed (and returned) to be zero.
    '''
    def __init__(self, mi=None):
        ''' 
        :param mi: Multi-index in tuple form
        :type mi: Optional. Either listing entries directly as in (7,8,0,9) or listing non-zero entries together with their dimension as in
        ((0,7),(1,8),(3,9))
        '''
        self.multiindex = dict()
        if mi:
            if type(mi[0]) == tuple:
                for dim, v in mi:
                    self.__setitem__(dim, v)
            else:
                for dim in range(len(mi)):
                    if mi[dim] > 0:
                        self.__setitem__(dim, mi[dim])
            
    def __getitem__(self, dim):
        if dim in self.multiindex.keys():
            return self.multiindex[dim]
        else:
            return 0
        
    def __setitem__(self, dim, value):
        if value > 0:
            self.multiindex[dim] = value
        else:
            if dim in self.multiindex.keys():
                self.multiindex.pop(dim)
                
    def sparse_tuple(self):
        '''
        Return non-zero entries together with their dimensions.
        E.g.: (3,2,0,0,1)-> ((0,3),(1,2),(5,1))
        
        :rtype: List of tuples
        '''
        # SORTING IS IMPORTANT. DONT REPLACE BY SOMETHING FASTER
        return tuple(sorted(self.multiindex.items()))
    
    def copy(self):
        '''
        Return deep copy
        '''
        A = SparseIndex()
        A.multiindex = self.multiindex.copy()
        return A
    
    def __hash__(self):
        return self.sparse_tuple().__hash__()
        
    def active_dims(self):
        '''
        Return dimensions with non-zero entries
        '''
        return list(self.multiindex.keys())
    
    def __eq__(self, other):
        return self.multiindex == other.multiindex
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def mod(self, mod):
        r'''
        Return copy with zeroes in dimensions specified by :code:`mod` (=modulo)
        
        :param mod: Dimensions to be ignored 
        :type mod: boolean function
        '''
        new = self.copy()
        for dim in list(new.multiindex.keys()):
            if mod(dim):
                new[dim] = 0
        return new
                
    def is_unit(self):
        '''
        Return whether multi-index is unit vector, i.e. has exactly one non-zero entry, which is one.
        '''
        return len(self.multiindex) == 1 and self.multiindex[list(self.multiindex.keys())[0]] == 1
        
    def equal_mod(self, other, mod):
        '''
        Compare multi-indices, ignoring dimensions in mod(=modulo)
        
        :param mod: Dimensions to be ignored
        :type mod: boolean function
        '''
        for dim in itertools.chain(self.multiindex.keys(), other.multiindex.keys()):
            if other[dim] != self[dim] and not mod(dim):
                return False
        return True
    
    def restrict(self, dimensions):  # opposite of .mod
        '''
        Return copy that has non-zero entries only in specified dimensions
        
        :param dimensions: Restrict to these entries
        :type dimensions: boolean function
        '''
        new = self.copy()
        for dim in list(new.multiindex.keys()):
            if not dimensions(dim):
                new[dim] = 0
        return new
    
    def __iter__(self):
        return iter(self.sparse_tuple()) 
    
    def __sub__(self, other):
        new = self.copy()
        for dim in other.multiindex:  # list(other.multiindex.keys()):
            new[dim] = new[dim] - other[dim]
        return new
    
    def __add__(self, other):
        new = self.copy()
        for dim in other.multiindex:  # list(other.multiindex.keys()):
            new[dim] = new[dim] + other[dim]
        return new
    
    def full_tuple(self, dim_max=None):
        '''
        Returns full representation, including non-zero entries up to specified maximal dimension
        
        :param dim_max: Optional. Maximal dimension to be included
        :rtype: tuple
        '''
        if dim_max is None:
            dim_max = 0 if len(self.active_dims()) == 0 else max(self.active_dims()) + 1
        return tuple((self[i] for i in range(dim_max)))
    
    def __str__(self):
        return self.full_tuple().__str__()
    
    def __repr__(self):
        return self.__str__()
    
    def retract(self, embed):
        '''
        Interpret entries in self as embedded entries of more compact multi-index and return the more compact multi-index.
        
        Assume we have an embedding:
        
        .. math::
        
           (a_0,a_1,a_2)\mapsto mi:=(a_0,0,a_1,0,a_2,0).
           
        :code:`mi.retract(embed)` allows to recover :math:`(a_0,a_1,a_2)` by specifying :code:`embed` as the embedding function
         :math:`0\mapsto 0, 1\mapsto 2, 2 \mapsto 4`
        
        :param embed: The embedding function
        :type embed: function
        :return: The retracted multi-index
        :rtype: SparseIndex
        '''
        midims = sorted(self.active_dims())
        i = 0
        cad = len(midims)
        dim = 0
        minew = SparseIndex()
        while i < cad:
            if embed(dim) == midims[i]:
                minew[dim] = self[midims[i]]
                i += 1
            dim += 1
        return minew    
    
    def rightshift(self, n=1):
        new = SparseIndex()
        for dim in self.multiindex:
            new.multiindex[dim + n] = self.multiindex[dim]
        return new
    
    def leftshift(self, n=1):
        new = SparseIndex()
        for dim in self.multiindex:
            if dim >= n:
                new.multiindex[dim - n] = self.multiindex[dim]
        return new
        
class SparseIndexDict(object): 
    '''
    Dictionary with SparseIndices as keys
    
    Only difference to usual dict is that dimensions in mod are automatically ignored when storing and accessing.
    
    :param mod: Dimensions that are ignored
    :type mod: :math:`\mathbb{N}\to\{\text{True},\text{False}\}`
    '''
    def __init__(self, mod=None, initializer=None):
        self.dict = {}
        self.mod = mod
        self.initializer = initializer
        
    def __contains__(self, si):
        if self.mod:
            si = si.mod(self.mod)
        return si in self.dict
    
    def __setitem__(self, si, value):
        if self.mod:
            si = si.mod(self.mod)
        self.dict[si] = value

    def __getitem__(self, si):
        if self.mod:
            si = si.mod(self.mod)
        if si in self.dict:
            return self.dict[si]
        else:
            if self.initializer:
                self.dict[si] = self.initializer(si)
                return self.dict[si]
            else:
                raise KeyError('Multi-index not contained in dictionary') 
        
    def __iter__(self):
        return self.dict.__iter__()
    
    def pop(self, si):
        if self.mod:
            si = si.mod(self.mod)
        self.dict.pop(si)
        
def get_admissible_indices(admissible, dim=-1):
    r'''
    Returns list :math:`\mathcal{I}:=\{\mathbf{k}: \verb|admissible|(\mathbf{k})=\text{True}\}`
    The admissibility condition must be downward closed, meaning that :math:`\mathbf{k}\in\mathcal{I}` implies
    :math:`\mathbf{\tilde{k}}\in\mathcal{I}` for any :math:`\tilde{\mathbf{k}}\leq \mathbf{k}` componentwise. 
    
    :param admissible: Decides if given multi-index is admissible
    :type admissible: :math:`\mathbb{N}^{n}\to\{\text{True, False}\}`
    :param dim: Maximal dimension. If not specified, dimensions are assumed to be 
    ordered with respect to admissibility of unit vectors (i.e. if one unit vector
     is not admissible, all subsequent ones are assumed to be neither)
    :return: :math:`\mathcal{I}:=\{\mathbf{k}: \verb|admissible|(\mathbf{k})=\text{True}\}`
    :rtype: List
    '''
    mis = []
    if admissible(SparseIndex()):
        mis.append(SparseIndex())
        while __next_admissible(admissible, mis[-1], dim):
            mis.append(__next_admissible(admissible, mis[-1], dim))
    return mis

def __next_admissible(admissible, mi, dim):
    if admissible(mi + SparseIndex((1,))):
        return mi + SparseIndex((1,))
    else:
        if dim == 1 or (dim < 1 and mi == SparseIndex()):
            return False
        else:
            tail = __next_admissible(lambda mi: admissible(mi.rightshift()), mi.leftshift(), dim - 1)
            if tail:
                return tail.rightshift();
            else:
                return False

def rectangle(L=None, c_dim=None):
    if not hasattr(L, '__contains__'):
        if not c_dim:
            raise ValueError('Specify either list of rectangle sides or c_dim')
        else:
            L = [L] * c_dim
    else:
        if c_dim:
            if c_dim != len(L):
                raise ValueError('c_dim does not match length of L')
        else:
            c_dim = len(L)
        
    def admissible(mi):
            return all([v < L[dim] for dim, v in mi])
    return get_admissible_indices(admissible, c_dim)
    
def pyramid(L, Gamma_1, Beta_1, Gamma_2, Beta_2, c_dim):
    def admissible(mi):
        return ((mi.leftshift() == SparseIndex() and (Gamma_1 + Beta_1) * mi[0] <= L) 
            or (mi.leftshift() != SparseIndex() and (Gamma_2 + Beta_2) * max([v for __, v in mi.leftshift()]) + (Gamma_1 + Beta_1) * mi[0] <= L))
    return get_admissible_indices(admissible, c_dim)
    
def hyperbolic_cross(L, exponents=None, c_dim=None):
    if not exponents:
        exponents = 1
    if not hasattr(exponents, '__contains__'):
        if not c_dim:
            raise ValueError('Specify either list of exponents or c_dim')
        else:
            exponents = [exponents] * c_dim
    else:
        if c_dim != len(exponents):
            raise ValueError('c_dim does not match length of L')
        else:
            c_dim = len(exponents)
    def admissible(mi):
        if exponents:
            return np.prod([(v + 1) ** exponents[i] for i, v in mi]) < L
        else: 
            return np.prod([v + 1 for __, v in mi]) < L
    return get_admissible_indices(admissible, c_dim)
    
def simplex(L, weights=None, c_dim=None):
    if not weights:
        weights = 1
    if not hasattr(weights, '__contains__'):
        if not c_dim:
            raise ValueError('Specify either list of rectangle sides or c_dim')
        else:
            weights = [weights] * c_dim
    else:
        if c_dim:
            if c_dim != len(weights):
                raise ValueError('c_dim does not match length of L')
        else:
            c_dim = len(weights)
    def admissible(mi):
            return sum([weights[dim] * v for dim, v in mi]) <= L
    return get_admissible_indices(admissible, c_dim)

def cartesian_product(entries, active_dims=None):
    '''
    Returns cartesian product of lists of reals.
    
    :param entries: Sets that are to be multiplied
    :type entries: List of iterables
    :param active_dims: If specified, empty sets are included in cartesian product, 
    and resulting multi-indices only have non-zero entries in active_dims
    :type active_dims: List of integers
    :return: Cartesian product
    :rtype: List of SparseIndices
    '''
    T = itertools.product(*entries)
    if active_dims:
        T = [SparseIndex(zip(active_dims, t)) for t in T]
    else:
        T = [SparseIndex(t) for t in T] 
    return T
