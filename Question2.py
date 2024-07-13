import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    X = np.array(vectors)
    means = np.mean(X, axis=1)
    X_centered = X - means[:, np.newaxis]
    cov_matrix = np.dot(X_centered, X_centered.T) / (X.shape[1] - 1)
    return cov_matrix.tolist()