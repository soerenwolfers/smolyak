import numpy as np
from smolyak.misc.default_dict import DefaultDict
from smolyak.misc.weighted_median import weighted_median

class Estimator(object):
    
    def __init__(self, dims_ignore, exponent_max, exponent_min, is_md=lambda dim: False):
        self.quantities = {}
        self.dims_ignore = dims_ignore
        self.ratios = DefaultDict(lambda dim: [])
        self.fallback_exponents = DefaultDict(lambda dim: 0)  # USED AS PRIOR IN EXPONENT ESTIMATION AND AS INITIAL GUESS OF EXPONENT WHEN NO DATA AVAILABLE AT ALL
        self.exponents = DefaultDict(lambda dim: self.fallback_exponents[dim])
        self.reliability = DefaultDict(lambda dim: 1)
        self.is_md = is_md
        self.exponent_max = exponent_max
        self.exponent_min = exponent_min
        self.FIT_WINDOW = np.Inf
        
    def set_fallback_exponent(self, dim, fallback_exponent):
        self.fallback_exponents[dim] = fallback_exponent
        
    def __contains__(self, mi):
        mi = mi.mod(self.dims_ignore)
        return mi in self.quantities
    
    def __setitem__(self, mi, q):
        mi = mi.mod(self.dims_ignore)
        self.quantities[mi] = q
        for dim in [dim for dim in mi.active_dims()]:
            mi_compare = mi.copy()
            mi_compare[dim] = mi_compare[dim] - 1
            if self.quantities[mi_compare] > 0:
                ratio_new = q / self.quantities[mi_compare]
                if self.is_md(dim) and mi_compare[dim] == 0:
                    ratio_new -= 1
                    if ratio_new < 0:
                        ratio_new = 0
            else:
                ratio_new = np.Inf  
            if len(self.ratios[dim]) < self.FIT_WINDOW:
                self.ratios[dim].append(ratio_new)
            else:
                self.ratios[dim] = self.ratios[dim][1:] + [ratio_new]
        self.__update_exponents()
        
    def __update_exponents(self):
        for dim in self.ratios:
            ratios = self.ratios[dim]
            estimate = max(min(np.median(ratios), np.exp(self.exponent_max)), np.exp(self.exponent_min))
            c = len(ratios)
            self.exponents[dim] = (self.fallback_exponents[dim] + c * np.log(estimate)) / (c + 1.)
            self.reliability[dim] = 1. / (1 + np.median([np.abs(ratio - estimate) for ratio in ratios]) / estimate)
            
    def __call__(self, mi):
        mi = mi.mod(self.dims_ignore)
        if mi in self.quantities:
            return self.quantities[mi]
        else:
            if mi.active_dims():
                q_neighbors = []
                w_neighbors = []
                for dim in mi.active_dims():
                    neighbor = mi.copy()
                    neighbor[dim] = neighbor[dim] - 1
                    q_neighbor = self.quantities[neighbor] * np.exp(self.exponents[dim])
                    q_neighbors.append(q_neighbor)
                    w_neighbors.append(self.reliability[dim])
                return weighted_median(q_neighbors, w_neighbors)
            else:
                return 1
