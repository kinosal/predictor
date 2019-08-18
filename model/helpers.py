import numpy as np


def mean_relative(y_pred, y_true):
    """
    Helper function to calculate mean relative deviation from two vectors
    = 1 - mean percentage error
    """
    return 1 - np.mean(np.abs((y_pred - y_true) / y_true))


def powerlist(start, base, times):
    """
    Helper function to create lists with exponential outputs,
    e.g. for search grids
    """
    array = []
    for i in range(0, times, 1):
        array.append(start * base ** i)
    return array
