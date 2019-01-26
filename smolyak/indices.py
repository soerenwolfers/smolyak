'''
Sparse multi-indices
'''
import itertools
import numpy as np
from swutil.collections import DefaultDict
from swutil.validation import Integer

class MultiIndex:
    '''
    Sparse multi-index representation.
    
    Dimension of multi-index is implicitly infinity, since non-set entries are assumed (and returned) to be zero.
    '''
    def __init__(self, mi=(),sparse=False):
        ''' 
        :param mi: Multi-index in tuple form
        :type mi: Either 'dense' format, such as (7,8,0,9), or sparse, i.e. 
        only non-zero entries together with their dimension as in ((0,7),(1,8),(3,9))
        :param sparse: Determines which of the formats above are used
        :type sparse: Boolean.
        '''
        self.multiindex = dict()
        if mi:
            if Integer.valid(mi):
                self[0] = mi
            else:
                for dim,v in mi if sparse else enumerate(mi):
                    self[dim]=v
    
    def sparse_tuple(self):
        '''
        Return non-zero entries together with their dimensions.
        E.g.: (3,2,0,0,1)-> ((0,3),(1,2),(5,1))
        
        :rtype: List of tuples
        '''
        return tuple(sorted(self.multiindex.items()))
    
    def copy(self):
        '''
        Return deep copy
        '''
        A = MultiIndex()
        A.multiindex = self.multiindex.copy()
        return A
    
    def active_dims(self):
        '''
        Return dimensions with non-zero entries
        '''
        return sorted(list(self.multiindex.keys()))
        
    def max_dim(self):
        '''
        Return maximal active dimension, e.g. (1,0,1) returns 3 and (1,0,0) returns 1 and (0,) returns 0
        '''
        return max(self.active_dims()) + 1 if self.multiindex else 0 

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
                
    def is_kronecker(self):
        '''
        Return whether multi-index is unit vector, i.e. has exactly one non-zero entry, which is one.
        '''
        return len(self.active_dims()) == 1 and self[self.active_dims()[0]] == 1
        
    def equal_mod(self, other, mod):
        '''
        Compare multi-indices, ignoring dimensions in mod(=modulo)
        
        :param mod: Dimensions to be ignored
        :type mod: boolean function
        '''
        for dim in set(self.multiindex.keys())|set(other.multiindex.keys()):
            if other[dim] != self[dim] and not mod(dim):
                return False
        return True

    def sum(self):
        return sum(self.multiindex.values())

    def restrict(self, dimensions):  
        '''
        Return copy that has non-zero entries only in specified dimensions
        mi.restrict(dimensions) is the same as mod(lambda d: not dimension(d)). 

        :param dimensions: Restrict to these entries
        :type dimensions: boolean function
        '''
        new = self.copy()
        for dim in new.active_dims():
            if not dimensions(dim):
                new[dim] = 0
        return new
    
    def full_tuple(self, c_dim=None):
        '''
        Returns full representation
        
        :param c_dim: Number of dimensions to be included
        :rtype: tuple
        '''
        if c_dim is None:
            c_dim = max(self.active_dims() or [-1])+1
        return tuple((self[i] for i in range(c_dim)))

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
        :rtype: MultiIndex
        '''
        midims = sorted(self.active_dims())
        i = 0
        cad = len(midims)
        dim = 0
        minew = MultiIndex()
        while i < cad:
            if embed(dim) == midims[i]:
                minew[dim] = self[midims[i]]
                i += 1
            dim += 1
        return minew    
    
    def shifted(self, n=1):
        '''
        Shift multi-index by n entries to the right
        '''
        new = MultiIndex()
        for dim in self.multiindex:
            if dim+n>=0:
                new.multiindex[dim + n] = self.multiindex[dim]
        return new
    
    def __lt__(self,other):
        d1,d2 = self.active_dims(),other.active_dims()
        if not (d1 or d2):
            return False
        else:
            dims = itertools.chain(d1,d2)  
            dim_max=max(dims)
            return np.all(np.less(self.full_tuple(dim_max+1),other.full_tuple(dim_max+1)))
    
    def __le__(self,other):
        d1,d2 = self.active_dims(),other.active_dims()
        if not (d1 or d2):
            return True
        else:
            dims = itertools.chain(d1,d2)  
            dim_max=max(dims)
            return np.all(np.less_equal(self.full_tuple(dim_max+1),other.full_tuple(dim_max+1)))
            
    def __eq__(self, other):
        return self.multiindex == other.multiindex
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return self.sparse_tuple().__hash__()
    
    def __iter__(self):
        return iter(self.sparse_tuple()) 
    
    def __sub__(self, other):
        new = self.copy()
        for dim in other.multiindex:
            new[dim] = new[dim] - other[dim]
        return new
    
    def __add__(self, other):
        new = self.copy()
        for dim in other.multiindex:
            new[dim] = new[dim] + other[dim]
        return new  

    # def __mul__(self,other):
    #     new = self.copy()
    #     for dim in new.multiindex:
    #         new[dim] = other*new[dim]
    #     return new  

    def __rmul__(self,other):
        new = self.copy()
        for dim in new.multiindex:
            new[dim] = other*new[dim]
        return new  
    
    def __radd__(self,other):
        if other ==0:
            return self.copy()
        else:
            raise ValueError('Cannot add MultiIndex to {}'.format(other))
        
    def __str__(self):
        return self.full_tuple().__str__()
    
    def __repr__(self):
        return self.__str__()  
        
    def __getitem__(self, dim):
        try:    
            return self.multiindex[dim]
        except:
            if isinstance(dim,int):
                return 0
            else:
                try: 
                    new = MultiIndex()
                    for i in dim:
                        new[i] = self[i]
                except:
                    new = MultiIndex()
                    for i in range(*dim.indices(self.max_dim())):
                        new[i] = self[i]
                return new
        
    def __setitem__(self, dim, value):
        if value != 0:
            self.multiindex[dim] = value
        else:
            if dim in self.multiindex.keys():
                self.multiindex.pop(dim)
    
    def __neg__(self):
        new  = self.copy()
        for dim in new.multiindex:
            new[dim] = -new[dim]
        return new

class MISet:
    '''
    Stores connected sets of multi-indices and corresponding 
    admissible multi-indices.
    '''
    def __init__(self,mis=(),dims = ()):
        self.mis = []
        self.active_dims = set()
        self.candidates = {MultiIndex()} 
        if Integer.valid(dims):
            dims = range(dims)
        self.add_dimensions(dims)
        self.update(mis)
    
    def add(self,mi):
        '''
        Add new multi-index. 
        
        :param mi: Multi-index to be added
        :type mi:  MultiIndex
        '''
        if not mi in self.mis:
            self.mis.append(mi)
            if mi in self.candidates:
                self.candidates -= {mi}
            else:
                raise ValueError('Multi-index is not admissible')
            new_candidates = (mi + sign*kronecker(dim) for dim,sign in itertools.product(self.active_dims,(-1,1)))
            self.candidates.update(mi for mi in new_candidates if (mi not in self.mis and mi>=MultiIndex())) 
    
    def update(self,mis):
        '''
        Add multiple new multi-indices.
        
        :param mis: Multi-indices to be added
        :type mis: Iterable of SparseIndices
        '''
        mis=sorted(mis)
        for mi in mis:
            self.add(mi)     
        
    def add_dimensions(self,dims):
        for dim in dims:
            if dim in self.active_dims:
                raise ValueError('{} was already an active dimension of downward closed set'.format(dim))
            else:
                self.active_dims.add(dim)
                if MultiIndex() in self.mis:
                    self.candidates.add(kronecker(dim))

    def is_admissible(self, mi):
        '''
        Check if given multi-index is admissible.
        Returns true also if multi-index is already in set!
        '''
        return (mi == MultiIndex()) or any(mi + sign*kronecker(dim) in self.mis for dim,sign in itertools.product(self.active_dims,(-1,1)))
        
    def __iter__(self):
        return self.mis.__iter__()
       
    def __str__(self):
        return list(self.mis).__str__()
               
    def __repr__(self):
        return 'MISet('+list(self.mis).__str__()+')'
                
    def __contains__(self,mi):
        return mi in self.mis
    
def kronecker(dim):
    '''
    Returns kronecker vector with single entry 1 in dimension :code:`dim` 
    '''
    mi = MultiIndex()
    mi[dim] = 1
    return mi
    
class MultiIndexDict: 
    '''
    Dictionary with SparseIndices as keys
    
    Only difference to usual dict is that dimensions in mod are automatically ignored when storing and accessing.
    
    :param mod: Dimensions that are ignored
    :type mod: :math:`\mathbb{N}\to\{\text{True},\text{False}\}`
    :param initializer: Default values
    :type initializer: Function
    '''
    def __init__(self, mod=None, initializer=None):
        self._dict = {}
        self.mod = mod
        self.initializer = initializer
    
    def pop(self, si):
        if self.mod:
            si = si.mod(self.mod)
        self._dict.pop(si)
        
    def __contains__(self, si):
        if self.mod:
            si = si.mod(self.mod)
        return si in self._dict
    
    def __setitem__(self, si, value):
        if self.mod:
            si = si.mod(self.mod)
        self._dict[si] = value

    def __getitem__(self, si):
        if self.mod:
            si = si.mod(self.mod)
        if si in self._dict:
            return self._dict[si]
        else:
            if self.initializer:
                self._dict[si] = self.initializer(si)
                return self._dict[si]
            else:
                raise KeyError('Multi-index not contained in dictionary') 
        
    def __iter__(self):
        return self._dict.__iter__()
    
    def __str__(self):
        return self._dict.__str__()
    
    def __repr__(self):
        return self.__str__()
 
def get_bundles(sparse_indices,is_bundled):
    '''
    Slice multi-index set into bundles
    '''
    bundles=DefaultDict(lambda _: list())
    not_bundled=lambda dim: not is_bundled(dim)
    for si in sparse_indices:
        representer=si.restrict(not_bundled)
        bundles[representer]+=[si]
    return list(bundles.values())


def get_bundle(multi_index,multi_indices,is_bundled):
    '''
    Return bundle of sparse_index within sparse_indices
    '''
    return [mi for mi in multi_indices
               if multi_index.equal_mod(mi,is_bundled)]
    
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
    if dim==0:
        if admissible(MultiIndex()):
            return [MultiIndex()]
        else:
            return []
    mis = []
    def next_admissible(mi, dim, shift = 0):
        test = (mi + kronecker(0)).shifted(shift)
        if admissible(test):
            return test 
        else:
            if dim == 1 or (dim < 1 and mi == MultiIndex()):
                return False
            else:
                return next_admissible(mi.shifted(-1), dim - 1,shift = shift+1)
    if admissible(MultiIndex()):
        mis.append(MultiIndex())
        while True:
            na = next_admissible(mis[-1], dim)
            if na:
                mis.append(na)
            else:
                break
    return mis
    
def rectangle(L=None, n=None):
    if not hasattr(L, '__contains__'):
        if n is None:
            raise ValueError('Specify either list of rectangle sides or n')
        else:
            L = [L] * n
    else:
        if n is not None:
            if n != len(L):
                raise ValueError('n does not match length of L')
        else:
            n = len(L)
        
    def admissible(mi):
            return all([v <= L[dim] for dim, v in mi])
    return get_admissible_indices(admissible, n)
    
def pyramid(L, Gamma_1, Beta_1, Gamma_2, Beta_2, n):
    def admissible(mi):
        return ((mi.leftshift() == MultiIndex() and (Gamma_1 + Beta_1) * mi[0] <= L) 
            or (mi.leftshift() != MultiIndex() and (Gamma_2 + Beta_2) * max([v for __, v in mi.leftshift()]) + (Gamma_1 + Beta_1) * mi[0] <= L))
    return get_admissible_indices(admissible, n)
    
def hyperbolic_cross(L, exponents=None, n=None):
    if not exponents:
        exponents = 1
    if not hasattr(exponents, '__contains__'):
        if not n:
            raise ValueError('Specify either list of exponents or n')
        else:
            exponents = [exponents] * n
    else:
        if n != len(exponents):
            raise ValueError('n does not match length of exponents')
        else:
            n = len(exponents)
    def admissible(mi):
        if exponents:
            return np.prod([(v + 1) ** exponents[i] for i, v in mi]) <= L+1
        else: 
            return np.prod([v + 1 for __, v in mi]) <= L+1
    return get_admissible_indices(admissible, n)
    
def simplex(L, weights=None, n=None):
    '''
    Returns n-dimensional simplex :math:`\{(k_1,\dots,k_n)\in\mathbb{N}^n : k\dot w \leq L\}`  
    '''
    check_shortcut = False
    if weights is None:
        weights = 1
        check_shortcut = True
    if not hasattr(weights, '__contains__'):
        if not n:
            raise ValueError('Specify either list of weights sides or n')
        else:
            weights = [weights] * n
    else:
        if n:
            if n != len(weights):
                raise ValueError('n does not match length of L')
        else:
            n = len(weights)
    if check_shortcut and L==1:
        return [MultiIndex()]+[kronecker(dim) for dim in range(n)]
    def admissible(mi):
            return sum([weights[dim] * v for dim, v in mi]) <= L
    return get_admissible_indices(admissible, n)

def cartesian_product(entries, dims=None):
    '''
    Returns cartesian product of lists of reals.
    
    :param entries: Lists of integers or reals that are to be multiplied
    :type entries: List of lists or List of iterables
    :param dims: If specified, empty sets are included in cartesian product, 
    and resulting multi-indices only have non-zero entries in dims
    :type dims: List of integers
    :return: Cartesian product
    :rtype: List of MultiIndices
    '''
    return [MultiIndex(zip(dims or range(len(entries)), t),sparse=True) for t in itertools.product(*entries)]

def tensor_product(sets,ns):
    shifts = [0,*np.cumsum(ns)]
    return [sum(mi.shifted(shifts[k]) for k,mi in enumerate(mis)) for mis in itertools.product(*sets)]

class MixedDifferences:
    r'''
    Provided a function :math:`f\colon\mathbb{N}^n \to Y`, instances of this class 
    represent the associated mixed differences
    :math:`\Delta_{\text{mix}} f\colon \mathbb{N}^n \to Y`
    '''
    def reset(self):
        if self.store_output:
            self.outputs = dict()
    
    def __init__(self, f, store_output=True,zipped=True,c_var=None,reparametrization=False):
        r'''
        :param f: :math:`f\colon \mathbb{N}^n \to Y` 
        f may return tuple (work_model,value), in which case also the work is kept track of
        :param store_output: Specifies whether calls to f should be cached
        :type store_output: Boolean.
        '''
        self.f = f
        self.c_var=c_var
        self.zipped=zipped
        self.reparametrization=reparametrization
        if not self.zipped:
            if not self.c_var:
                raise ValueError('Must specify number of variables')
            if self.reparametrization:
                self.f=lambda mi: f(*MultiIndex([2**v for v in mi.full_tuple(c_dim=self.c_var)]).full_tuple(c_dim=self.c_var))
            else:
                self.f=lambda mi: f(*mi.full_tuple(c_dim=self.c_var))
        self.store_output = store_output
        if self.store_output:
            self.outputs = dict()
    
    def __call__(self, mi):
        r'''
        Compute mixed difference
        
        :param mi: Input :math:`\mathbf{k}` of mixed difference
        :return: :math:`(\Delta_{\text{mix}} f)(\mathbf{k})` or tuple (work,:math:`(\Delta_{\text{mix}} f)(\mathbf{k})`) if work is kept track of
        '''
        y = []
        track_work = False
        total_work = 0
        dims=mi.active_dims()
        for down in cartesian_product([[0, 1]] * len(dims), dims):
            tmi = mi - down
            if not self.store_output or tmi not in self.outputs:
                output=self.f(tmi)
                if self.store_output:
                    self.outputs[tmi] = output
                found = False
            else:
                output = self.outputs[tmi]
                found = True
            if isinstance(output, tuple):
                track_work = True
                work = output[0]
                if not found:
                    total_work += work
                tv = output[1]
            else:
                tv = output
            y.append((-1) ** len(down.active_dims()) * tv)      
        if track_work:
            return (total_work, sum(y)) 
        else:
            return sum(y)

def combination_rule(mis):
    '''
    Compute coefficients of combination rule.
    
    :param mis: Multi-index set
    :type mis: List of SparseIndices
    :return: Non-zero coefficients of combination rule
    :rtype: Dictionary with multi-indices as keys and coefficients as values
    '''
    coefficients = DefaultDict(lambda c: 0)
    for mi in mis:
        dims = mi.active_dims()
        for down in cartesian_product([[0, 1]] * len(dims), dims):
            mi_neighbor = mi - down
            coefficients[mi_neighbor] = coefficients[mi_neighbor] + (-1) ** len(down.active_dims())
    output={}
    for mi in coefficients.keys():
        if abs(coefficients[mi]) > 0.1: 
            output[mi]=coefficients[mi]
    return output
