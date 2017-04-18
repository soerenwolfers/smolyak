'''
Weighted median
'''
import numpy as np
def weighted_median(values, weights):
    '''
    Returns element such that sum of weights below and above are (roughly) equal
    
    :param values: Values whose median is sought
    :type values: List of reals
    :param weights: Weights of each value
    :type weights: List of positive reals
    :return: value of weighted median
    :rtype: Real
    '''
    if len(values) == 1:
        return values[0]
    if len(values) == 0:
        raise ValueError('Cannot take median of empty list')
    values = [float(value) for value in values]
    indices_sorted = np.argsort(values)
    values = [values[ind] for ind in indices_sorted]
    weights = [weights[ind] for ind in indices_sorted]
    total_weight = sum(weights)
    below_weight = 0
    i = -1
    while below_weight < total_weight / 2:
        i += 1
        below_weight += weights[i]
    return values[i]

if __name__ == '__main__':
    print(weighted_median([1, 2, 33, 3], [1, 1, 1, 1])) 
