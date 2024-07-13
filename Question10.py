"""
Cross-Validation Data Split Implementation (medium)
Write a Python function that performs k-fold cross-validation data splitting from scratch.
The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature)
and an integer k representing the number of folds. 
The function should split the dataset into k parts, systematically use one part as the test set and the remaining as the training set,
and return a list where each element is a tuple containing the training set and test set for each fold.
"""

import numpy as np

import numpy as np

def cross_validation_split(data, k):
    np.random.shuffle(data)  # This line can be removed if shuffling is not desired in examples
    fold_size = len(data) // k
    folds = []
    
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size if i != k-1 else len(data)
        test = data[start:end]
        train = np.concatenate([data[:start], data[end:]])
        folds.append([train.tolist(), test.tolist()])
    
    return folds