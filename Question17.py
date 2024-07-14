"""
Single Neuron (easy)
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, 
handling multidimensional input features. The function should take a list of feature vectors (each vector representing multiple features for an example),
associated true binary labels, and the neuron's weights (one for each feature) and bias as input. 
It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels,
both rounded to four decimal places.
"""

import math

def single_neuron_model(features: list[list[float]], labels: list[float], weights: list[float], bias: float) -> tuple[list[float], float]:
    def sigmoid(z: float) -> float:
        return 1 / (1 + math.exp(-z))
    probabilities = [
        round(sigmoid(sum(w * f for w, f in zip(weights, feature)) + bias), 4)
        for feature in features
    ]
    mse = round(sum((p - l) ** 2 for p, l in zip(probabilities, labels)) / len(labels), 4)
    
    return probabilities, mse