# Activation-Functions-plain-python
Create Tanh, ReLU and Leaky ReLU activation functions by their formulas.
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


# Activation Functions in Python

## Rectified Linear Unit (ReLU) Activation Function

The Rectified Linear Unit, or ReLU, is a widely used activation function in neural networks. It introduces non-linearity by outputting the input directly if it is positive; otherwise, it returns zero. The ReLU function is defined by the formula:

\[ \text{ReLU}(x) = \max(0, x) \]

### Code

```python
def relu(x):
    return max(0, x)

# Example usage:
result1 = relu(-100)  # Returns: 0
result2 = relu(8)     # Returns: 8
