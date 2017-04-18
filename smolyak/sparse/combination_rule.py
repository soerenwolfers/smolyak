from smolyak.sparse.sparse_index import cartesian_product
from smolyak.misc.default_dict import DefaultDict

def combination_rule(mis, function=None):
    '''
    Compute coefficients of combination rule.
    
    :param mis: Multi-index set
    :type mis: List of SparseIndices
    :return: Non-zero coefficients of combination rule
    :rtype: Dictionary with multi-indices as keys and coefficients as values
    '''
    coefficients = DefaultDict(lambda c: 0)
    for mi in mis:
        active_dims = mi.active_dims()
        downs = cartesian_product([[0, 1]] * len(active_dims), active_dims=active_dims)
        for down in downs:
            mi_neighbor = mi - down
            coefficients[mi_neighbor] = coefficients[mi_neighbor] + (-1) ** sum(down.multiindex.values())
    for mi in coefficients.keys():
        if abs(coefficients[mi]) < 0.1: 
            coefficients.pop(mi)
    return coefficients
