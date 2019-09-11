import numpy as np


def mean_relative_accuracy(y_pred, y_true):
    """
    Helper function to calculate mean relative closeness of two vectors
    = 1 - mean percentage error
    """
    return 1 - np.mean(np.abs((y_pred - y_true) / y_true))


def powerlist(start, base, times):
    """
    Helper function to create lists with exponential outputs,
    e.g. for search grids
    """
    return [start * base ** i for i in range(0, times, 1)]
