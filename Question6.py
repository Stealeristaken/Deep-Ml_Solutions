"""
Linear Regression Using Normal Equation (easy)
Write a Python function that performs linear regression using the normal equation.
The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. 
Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.
"""

import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X, y = np.array(X), np.array(y)
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    return np.round(theta, 4).tolist()