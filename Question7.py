"""
Linear Regression Using Gradient Descent (easy)
Write a Python function that performs linear regression using gradient descent. 
The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input,
along with learning rate alpha and the number of iterations, and return the coefficients of the linear regression model as a NumPy array. 
Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.
"""

import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    theta = np.zeros(X.shape[1])
    for _ in range(iterations):
        theta -= alpha * (X.T @ (X @ theta - y)) / len(y)
    return np.round(theta, 4)