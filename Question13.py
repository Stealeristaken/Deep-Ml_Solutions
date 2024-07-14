"""
Decision Tree Learning (hard)
Write a Python function that implements the decision tree learning algorithm for classification. 
The function should use recursive binary splitting based on entropy and information gain to build a decision tree.
It should take a list of examples (each example is a dict of attribute-value pairs) and a list of attribute names as input,
and return a nested dictionary representing the decision tree.
"""

import math
from collections import Counter
from typing import Dict, List, Any

def entropy(labels: List[Any]) -> float:
    counts = Counter(labels)
    return -sum((count / len(labels)) * math.log2(count / len(labels)) for count in counts.values())

def information_gain(examples: List[Dict[str, Any]], attr: str, target: str) -> float:
    total_entropy = entropy([example[target] for example in examples])
    weighted_attr_entropy = sum(
        (len(subset := [e for e in examples if e[attr] == value]) / len(examples)) * entropy([e[target] for e in subset])
        for value in set(example[attr] for example in examples)
    )
    return total_entropy - weighted_attr_entropy

def decision_tree(examples: List[Dict[str, Any]], attributes: List[str], target: str) -> Any:
    if not examples:
        return "No examples"
    if len(set(example[target] for example in examples)) == 1:
        return examples[0][target]
    if not attributes:
        return Counter(example[target] for example in examples).most_common(1)[0][0]

    best_attr = max(attributes, key=lambda attr: information_gain(examples, attr, target))
    tree = {best_attr: {}}

    for value in set(example[best_attr] for example in examples):
        subset = [example for example in examples if example[best_attr] == value]
        subtree = decision_tree(subset, [attr for attr in attributes if attr != best_attr], target)
        tree[best_attr][value] = subtree

    return tree