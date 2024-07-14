"""
Softmax Activation Function Implementation (easy)
Write a Python function that computes the softmax activation for a given list of scores.
The function should return the softmax values as a list, each rounded to four decimal places.
"""
import math

def softmax(scores: list[float]) -> list[float]:
    exp_scores = [math.exp(score - max(scores)) for score in scores]
    return [round(score / sum(exp_scores), 4) for score in exp_scores]