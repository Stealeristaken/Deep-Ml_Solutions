""""
Determinant of a 4x4 Matrix using Laplace's Expansion (hard)
Write a Python function that calculates the determinant of a 4x4 matrix using Laplace's Expansion method. 
The function should take a single argument, a 4x4 matrix represented as a list of lists, and return the determinant of the matrix. 
The elements of the matrix can be integers or floating-point numbers. 
Implement the function recursively to handle the computation of determinants for the 3x3 minor matrices.
"""

def determinant_4x4(matrix: list[list[int|float]]) -> float:
    def det_3x3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    det = 0
    for i in range(4):
        minor = [row[:i] + row[i+1:] for row in matrix[1:]]
        det += (-1)**i * matrix[0][i] * det_3x3(minor)
    
    return det