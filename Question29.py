"""
Write a Python function to perform a random shuffle of the samples in two numpy arrays,
X and y, while maintaining the corresponding order between them. 
The function should have an optional seed parameter for reproducibility.
"""

import numpy as np

def shuffle_data(X, y, seed=None):
	if seed:
		np.random.seed(seed)
		
	idx = np.arange(X.shape[0])
	np.random.shuffle(idx)
	return X[idx], y[idx]
