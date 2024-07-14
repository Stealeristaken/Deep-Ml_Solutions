"""
Principal Component Analysis (PCA) Implementation (medium)
Write a Python function that performs Principal Component Analysis (PCA) from scratch.
The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature.
The function should standardize the dataset, compute the covariance matrix, find the eigenvalues and eigenvectors,
and return the principal components (the eigenvectors corresponding to the largest eigenvalues). 
The function should also take an integer k as input, representing the number of principal components to return.
"""

import numpy as np

def pca(data: np.ndarray, k: int) -> list[list[int|float]]:
    X_std = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    cov_matrix = np.cov(X_std.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    principal_components = eigenvectors[:, :k].tolist()
    
    return principal_components