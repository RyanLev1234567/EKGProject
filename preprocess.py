import numpy as np
from scipy.signal import resample

def normalize_signal(X):
    """Normalize each lead to mean 0, std 1"""
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

def resample_signal(X, target_len=1000):
    """Resample signals to fixed length"""
    return resample(X, target_len)
