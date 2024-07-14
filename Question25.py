"""
Calculate 2x2 Matrix Inverse (medium)
Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.
"""

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:

    [[a, b], [c, d]] = matrix
    det = a * d - b * c
    if abs(det) < 1e-10:  # Use a small threshold instead of exact zero
        return None
    inv_det = 1 / det
    return [[round(d * inv_det, 4), round(-b * inv_det, 4)],
            [round(-c * inv_det, 4), round(a * inv_det, 4)]]