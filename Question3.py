"""
Solve Linear Equations using Jacobi Method (medium)

Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b.
The function should iterate 10 times, rounding each intermediate solution to four decimal places, and return the approximate solution x.
"""
import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros_like(b)
    D = np.diag(A)
    R = A - np.diag(D)
    for _ in range(n):
        x = (b - np.dot(R, x)) / D
    return np.round(x, 4).tolist()