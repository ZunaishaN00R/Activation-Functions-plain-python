# Activation-Functions-plain-python

Create Tanh, ReLU, and Leaky ReLU activation functions in plain Python.

# Activation Functions in Python

## Hyperbolic Tangent (Tanh) Activation Function

The hyperbolic tangent, or Tanh, activation function is commonly used in neural networks to introduce non-linearity. The Tanh function is defined by the formula:

\[ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \]

### Code

```python
import math

def hyperbolic(z):
    x = math.exp(z) - math.exp(-z)
    y = math.exp(z) + math.exp(-z)
    tanh = x / y
    return tanh

# Example usage:
result = hyperbolic(-56)  # Returns: -1.0
