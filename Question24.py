"""
Matrix Transformation (medium)
Write a Python function that transforms a given matrix A using the operation 
T^-1AS, where T and S are invertible matrices. The function should first validate if the matrices T and S are invertible, and then perform the transformation
"""

import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    A = np.array(A)
    T = np.array(T)
    S = np.array(S)
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        raise ValueError("The matrices T and/or S are not invertible.")
    T_inv = np.linalg.inv(T)
    transformed_matrix = T_inv.dot(A).dot(S)
    return transformed_matrix.tolist()