"""
Calculate Mean by Row or Column (easy)
Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. 
The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.
"""

import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    arr = np.array(matrix)
    axis = 0 if mode == 'column' else 1
    return np.mean(arr, axis=axis).round(4).tolist()