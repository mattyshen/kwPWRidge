import numpy as np

def exp_kernel(center, bandwidth, X):
    return np.exp(-((X - center) ** 2) / (2 * bandwidth ** 2))

def abs_kernel(center, bandwidth, X):
    return np.exp(-(np.abs(X - center)) / (2 * bandwidth ** 2))

