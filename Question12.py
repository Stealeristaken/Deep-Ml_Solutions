"""
Transpose of a Matrix (easy)
Write a Python function that computes the transpose of a given matrix.
"""

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return [list(t) for t in zip(*a)]