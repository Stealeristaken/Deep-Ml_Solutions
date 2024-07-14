"""
Calculate Eigenvalues of a Matrix (medium)
Write a Python function that calculates the eigenvalues of a 2x2 matrix. 
The function should return a list containing the eigenvalues, sort values from highest to lowest.
"""

import math

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    determinant = a * d - b * c
    discriminant = math.sqrt(trace**2 - 4*determinant)
    lambda1 = (trace + discriminant) / 2
    lambda2 = (trace - discriminant) / 2
    return sorted([lambda1, lambda2], reverse=True)