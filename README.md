# Activation-Functions-plain-python

Create Tanh, ReLU, and Leaky ReLU activation functions in plain Python.

# Activation Functions in Python

## Hyperbolic Tangent (Tanh) Activation Function

The hyperbolic tangent, or Tanh, activation function is commonly used in neural networks to introduce non-linearity. The Tanh function is defined by the formula:

\[ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \]

Rectified Linear Unit (ReLU) Activation Function
The Rectified Linear Unit, or ReLU, is a widely used activation function in neural networks. It introduces non-linearity by outputting the input directly if it is positive; otherwise, it returns zero. The ReLU function is defined by the formula:

ReLU
(
�
)
=
max
⁡
(
0
,
�
)
ReLU(x)=max(0,x)

Code
python
Copy code
def relu(x):
    return max(0, x)

# Example usage:
result1 = relu(-100)  # Returns: 0
result2 = relu(8)     # Returns: 8
Leaky Rectified Linear Unit (Leaky ReLU) Activation Function
The Leaky Rectified Linear Unit, or Leaky ReLU, is a modification of the ReLU activation function. It allows a small, non-zero gradient when the input is negative, addressing the "dying ReLU" problem.

Code
python
Copy code
def leaky_relu(x):
    return max(0.1 * x, x)

# Example usage:
result = leaky_relu(-100)  # Returns: -10.0
Feel free to adjust this to fit your preferences and specific documentation style.

