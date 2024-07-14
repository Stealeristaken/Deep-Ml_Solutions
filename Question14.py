"""
Pegasos Kernel SVM Implementation (advanced)
Write a Python function that implements the Pegasos algorithm to train a kernel SVM classifier from scratch.
The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature), 
a label vector (1D NumPy array where each entry corresponds to the label of the sample), 
and training parameters such as the choice of kernel (linear or RBF), regularization parameter (lambda), and the number of iterations. 
The function should perform binary classification and return the model's alpha coefficients and bias.
"""

import numpy as np
from functools import partial

def linear_kernel(x, y):
    return np.dot(x, y)

def rbf_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * sigma**2))

def pegasos_kernel_svm(data, labels, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0):
    """
    Implements the Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm 
    for kernel Support Vector Machines.

    This function uses a stochastic subgradient method to solve the optimization 
    problem cast by Support Vector Machines. It supports both linear and RBF kernels.

    Parameters:
    data (array-like): The input features.
    labels (array-like): The corresponding labels.
    kernel (str): The type of kernel to use ('linear' or 'rbf'). Default is 'linear'.
    lambda_val (float): The regularization parameter. Default is 0.01.
    iterations (int): The number of iterations to run the algorithm. Default is 100.
    sigma (float): The sigma parameter for the RBF kernel. Default is 1.0.

    Algorithm steps:
    1. Initialize the dual variables (alphas) and the bias term (b).
    2. Select the appropriate kernel function.
    3. For each iteration and each data point:
       - Compute the decision function.
       - If the point is misclassified or within the margin, update its alpha and the bias.
    4. The algorithm uses a learning rate that decreases over time: η = 1 / (λt).
    5. The decision function is computed as a weighted sum of kernel evaluations.

    Returns:
    tuple: A tuple containing:
        - alphas (list): The dual variables, rounded to 4 decimal places.
        - b (float): The bias term, rounded to 4 decimal places.
    This implementation is efficient due to its use of vectorized operations and its 
    ability to work with both linear and RBF kernels. The Pegasos algorithm is known 
    for its fast convergence properties, making it suitable for large-scale learning tasks.
    """

    n_samples = len(data)
    alphas = np.zeros(n_samples)
    b = 0
    
    kernel_func = linear_kernel if kernel == 'linear' else partial(rbf_kernel, sigma=sigma)
    
    for t in range(1, iterations + 1):
        eta = 1.0 / (lambda_val * t)
        for i in range(n_samples):
            decision = np.sum(alphas * labels * [kernel_func(data[j], data[i]) for j in range(n_samples)]) + b
            if labels[i] * decision < 1:
                alphas[i] += eta * (labels[i] - lambda_val * alphas[i])
                b += eta * labels[i]

    return alphas.round(4).tolist(), round(b, 4)