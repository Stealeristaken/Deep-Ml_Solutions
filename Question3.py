import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros_like(b)
    D = np.diag(A)
    R = A - np.diag(D)
    for _ in range(n):
        x = (b - np.dot(R, x)) / D
    return np.round(x, 4).tolist()