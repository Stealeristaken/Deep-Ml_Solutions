"""
Given basis vectors in two different bases B and C for R^3, write a Python function to compute the transformation matrix P from basis B to C.
"""
import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	C = np.array(C)
	B = np.array(B)
	
	C_inv = np.linalg.inv(C)
	
	P = np.dot(C_inv, B)
	P = P.tolist()
	return P