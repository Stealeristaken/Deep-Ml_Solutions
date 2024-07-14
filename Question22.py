"""
Scalar Multiplication of a Matrix (easy)
Write a Python function that multiplies a matrix by a scalar and returns the result.
"""

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    return list(map(lambda row: list(map(lambda x: x * scalar, row)), matrix))