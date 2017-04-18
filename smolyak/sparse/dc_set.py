'''
Downward closed sets
'''
class DCSet(object):
    '''
    Stores downward closed sets of multi-indices.    
    '''
    def __init__(self):
        self.mis = []
        self.active_dims = set()
    
    def add_mi(self, mis):
        '''
        Add new multi-index and returns newly admissible multi-indices.
        
        A multi-index is called admissible here if the set of multi-indices remains downward closed.
        
        :param mis: (List of) multi-index to be added
        :type mi: (List of) SparseIndex
        :return: Multi-indices that become admissible after :code:`mi` is added.
        :rtype: Set of SparseIndex
        '''
        if not hasattr(mis, '__contains__'):
            mis = [mis]
        new_candidates = set()
        for mi in mis:
            self.mis.append(mi)
            if mi in new_candidates:
                new_candidates -= {mi}
            self.active_dims |= set(mi.active_dims())
            for dim in self.active_dims:
                candidate = mi.copy()
                candidate[dim] = candidate[dim] + 1
                if self.is_admissible(candidate):
                    new_candidates |= {candidate}
        return new_candidates
    
    def __iter__(self):
        return self.mis.__iter__()
                
    def is_admissible(self, mi):
        '''
        Check if given multi-index is admissible.
        '''
        for dim in mi.active_dims():
            test = mi.copy()
            test[dim] -= 1
            if not test in self.mis:
                return False
        return True
