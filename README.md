# Activation-Functions-plain-python

Create Tanh, ReLU, and Leaky ReLU activation functions in plain Python.

## Activation Functions

This repository provides implementations of three popular activation functions in machine learning: Hyperbolic Tangent (Tanh), Rectified Linear Unit (ReLU), and Leaky Rectified Linear Unit (Leaky ReLU).

### Hyperbolic Tangent (Tanh)

The hyperbolic tangent function, or Tanh, squashes input values to the range [-1, 1]. It is often used in neural networks to introduce non-linearity and is particularly useful in scenarios where the output should be centered around zero.

### Rectified Linear Unit (ReLU)

ReLU is a widely used activation function that returns the input for positive values and zero for negative values. It introduces non-linearity and has been successful in training deep neural networks.

### Leaky Rectified Linear Unit (Leaky ReLU)

Leaky ReLU is a variant of ReLU that allows a small, non-zero gradient when the input is negative.This helps mitigate the "dying ReLU" problem and can be useful in situations where some information from negative values is valuable.

