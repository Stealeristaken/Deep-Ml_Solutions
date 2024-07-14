"""
Implementing Basic Autograd Operations (medium)
Special thanks to Andrej Karpathy for making a video about this, if you haven't already check out his videos on YouTube https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg.
Write a Python class similar to the provided 'Value' class that implements the basic autograd operations: addition, multiplication, and ReLU activation. 
The class should handle scalar values and should correctly compute gradients for these operations through automatic differentiation.
"""

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        return self._operation(other, lambda x, y: x + y, '+')

    def __mul__(self, other):
        return self._operation(other, lambda x, y: x * y, '*')

    def relu(self):
        return self._operation(None, lambda x, _: max(0, x), 'ReLU')

    def _operation(self, other, forward_func, op):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(forward_func(self.data, other.data), (self, other), op)

        def _backward():
            if op == '+':
                self.grad += out.grad
                other.grad += out.grad
            elif op == '*':
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            elif op == 'ReLU':
                self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"