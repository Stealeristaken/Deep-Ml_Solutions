"""
Feature Scaling Implementation (easy)
Write a Python function that performs feature scaling on a dataset using both standardization and min-max normalization. 
The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. 
It should return two 2D NumPy arrays: one scaled by standardization and one by min-max normalization.
Make sure all results are rounded to the nearest 4th decimal.
"""

import numpy as np 

def feature_scaling(data):
    data = np.array(data)
    standardized = (data - data.mean(axis=0)) / data.std(axis=0)
    normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return np.round(standardized, 4).tolist(), np.round(normalized, 4).tolist()
