# Activation-Functions-plain-python

Create Tanh, ReLU, and Leaky ReLU activation functions in plain Python.

# Activation Functions

This repository provides implementations of three popular activation functions in machine learning: Hyperbolic Tangent (Tanh), Rectified Linear Unit (ReLU), and Leaky Rectified Linear Unit (Leaky ReLU).

## Hyperbolic Tangent (Tanh)

The hyperbolic tangent function, or Tanh, squashes input values to the range [-1, 1]. It is often used in neural networks to introduce non-linearity and is particularly useful in scenarios where the output should be centered around zero.

### Formula:
```python
import math

def hyperbolic_tangent(z):
    x = math.exp(z) - math.exp(-z)
    y = math.exp(z) + math.exp(-z)
    tanh = x / y
    return tanh
