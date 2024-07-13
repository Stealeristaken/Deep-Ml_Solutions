"""
Singular Value Decomposition (SVD) (hard)
Write a Python function that approximates the Singular Value Decomposition on a 2x2 matrix by using 
the jacobian method and without using numpy svd function,
i mean you could but you wouldn't learn anything. return the result in this format.
"""


import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A_T_A = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eigh(A_T_A)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    singular_values = np.sqrt(eigenvalues[sorted_indices])
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvectors, singular_values, eigenvectors.T