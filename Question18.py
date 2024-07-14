"""
Single Neuron with Backpropagation (medium)
Write a Python function that simulates a single neuron with sigmoid activation, 
and implements backpropagation to update the neuron's weights and bias.
The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs.
The function should update the weights and bias using gradient descent based on the MSE loss, and 
return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    weights = np.array(initial_weights)
    bias = initial_bias
    features = np.array(features)
    labels = np.array(labels)
    
    def forward(X):
        return sigmoid(np.dot(X, weights) + bias)
    
    def backward(X, y, y_pred):
        errors = y_pred - y
        grad = errors * y_pred * (1 - y_pred)
        return np.dot(X.T, grad), np.sum(grad)
    
    mse_values = []
    for _ in range(epochs):
        predictions = forward(features)
        mse_values.append(round(np.mean((predictions - labels) ** 2), 4))
        
        weight_grad, bias_grad = backward(features, labels, predictions)
        
        weights -= learning_rate * weight_grad / len(labels)
        bias -= learning_rate * bias_grad / len(labels)
    
    return np.round(weights, 4).tolist(), round(bias, 4), mse_values