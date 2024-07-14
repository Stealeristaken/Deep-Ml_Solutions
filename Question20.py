"""
Reshape Matrix (easy)
Write a Python function that reshapes a given matrix into a specified shape.
"""

import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	shaped_matrix = np.array(a).reshape(new_shape).tolist()
	return shaped_matrix