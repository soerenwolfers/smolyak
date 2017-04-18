'''
Mixed differences
'''
import itertools
from smolyak.sparse.sparse_index import SparseIndex

class MixedDifferences(object):
    r'''
    Provided a function :math:`f\colon\mathbb{N}^n \to Y`, instances of this class 
    represent the associated first order mixed differences
    :math:`\Delta_{\text{mix}} f\colon \mathbb{N}^n \to Y`
    '''
    def __init__(self, f, store_output=True):
        r'''
        :param f: :math:`f\colon \mathbb{N}^n \to Y` 
        f may return tuples (work_model,value), in which case also the work is kept track of
        :param store_output: Specifies whether calls to f should be cached
        :type store_output: Optional. Boolean.
        '''
        self.f = f;
        self.store_output = store_output
        if self.store_output:
            self.outputs = dict()
    
    def __call__(self, mi):
        r'''
        Compute mixed difference
        
        :param mi: Input :math:`\mathbf{k}` of mixed difference
        :return: :math:`(\Delta_{\text{mix}} f)(\mathbf{k})` or tuple (work,:math:`(\Delta_{\text{mix}} f)(\mathbf{k})`) if work is kept track of
        '''
        y = list()
        track_work = False
        total_work = 0
        for e in itertools.product(range(2), repeat=len(mi.active_dims())):
            tmi = mi - SparseIndex(list(zip(mi.active_dims(), e)))
            if not self.store_output or tmi not in self.outputs:
                output = self.f(tmi)
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
            y.append((-1) ** sum(e) * tv)      
        if track_work:
            return (total_work, sum(y)) 
        else:
            return sum(y)
        
    def reset(self):
        if self.store_output:
            self.outputs = dict()
