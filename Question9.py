"""
K-Means Clustering (medium)
Write a Python function that implements the k-Means algorithm for clustering, starting with specified initial centroids and a set number of iterations.
The function should take a list of points (each represented as a tuple of coordinates), an integer k representing the number of clusters to form, 
a list of initial centroids (each a tuple of coordinates), and an integer representing the maximum number of iterations to perform. 
The function will iteratively assign each point to the nearest centroid and update the centroids 
based on the assignments until the centroids do not change significantly,or the maximum number of iterations is reached. 
The function should return a list of the final centroids of the clusters.
Round to the nearest fourth decimal.
"""

import numpy as np

def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.array(points)
    centroids = np.array(initial_centroids)
    
    for _ in range(max_iterations):
        distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([points[labels == i].mean(axis=0) for i in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return [tuple(np.round(centroid, 4)) for centroid in centroids]