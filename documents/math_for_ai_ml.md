# Mathematics for AI & Machine Learning

## A Beginner's Friendly Guide

---

## Introduction

Welcome! This guide will teach you the math you need for AI and Machine Learning in a simple, easy-to-understand way. No prior math knowledge is assumed - we'll start from the basics.

**Why Math for AI/ML?**

- AI/ML algorithms are built on mathematical principles
- Understanding math helps you choose the right algorithms
- Debugging models requires mathematical intuition
- Advanced research needs strong math foundations

---

## Table of Contents

1. [Basic Arithmetic & Algebra](#1-basic-arithmetic--algebra)
2. [Functions and Graphs](#2-functions-and-graphs)
3. [Linear Algebra](#3-linear-algebra)
4. [Calculus Basics](#4-calculus-basics)
5. [Probability & Statistics](#5-probability--statistics)
6. [Optimization](#6-optimization)
7. [Summary & Next Steps](#7-summary--next-steps)

---

## 1. Basic Arithmetic & Algebra

### 1.1 Variables and Expressions

In math and programming, we use **variables** as placeholders for numbers:

```python
# Think of variables as boxes that hold numbers
x = 5      # x is a variable with value 5
y = 3      # y is a variable with value 3

# We can do operations with variables
sum = x + y       # Addition: 5 + 3 = 8
product = x * y  # Multiplication: 5 * 3 = 15
power = x ** y    # Power: 5^3 = 125
```

### 1.2 Equations

An **equation** says two things are equal:

```python
# Linear equation: y = mx + b
# This is the equation of a straight line!

m = 2    # slope (how steep the line is)
b = 3    # y-intercept (where line crosses y-axis)

# If x = 4:
# y = 2(4) + 3 = 11
```

### 1.3 Important Algebra Concepts

**Polynomials** (expressions with variables raised to powers):

```python
# Linear: ax + b (degree 1)
# Quadratic: ax² + bx + c (degree 2)
# Cubic: ax³ + bx² + cx + d (degree 3)

# Example: f(x) = 2x² + 3x + 1
# If x = 2: f(2) = 2(4) + 3(2) + 1 = 8 + 6 + 1 = 15
```

**Factoring** (breaking into smaller parts):

```python
# x² - 9 = (x + 3)(x - 3)  # Difference of squares
# x² + 5x + 6 = (x + 2)(x + 3)
```

### 1.4 Summations (Σ)

The sigma symbol Σ means "sum of":

```
∑ᵢ₌₁ⁿ i = 1 + 2 + 3 + ... + n

Example: ∑ᵢ₌₁⁵ i² = 1² + 2² + 3² + 4² + 5² = 1 + 4 + 9 + 16 + 25 = 55
```

In Python:

```python
total = sum(i**2 for i in range(1, 6))  # Result: 55
```

---

## 2. Functions and Graphs

### 2.1 What is a Function?

A **function** is like a machine: you put something in, and it gives something out.

```
f(x) = x² + 2

Input: 3 → Output: 11
Input: 0 → Output: 2
Input: -1 → Output: 3
```

### 2.2 Common Functions You Need to Know

#### Linear Function

```
f(x) = mx + b
```

- Straight line
- m = slope, b = intercept

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
y = 2 * x + 1  # f(x) = 2x + 1

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Function: y = 2x + 1')
plt.grid(True)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()
```

#### Quadratic Function

```
f(x) = ax² + bx + c
```

- Parabola (U-shaped curve)

```python
y = x**2  # f(x) = x²
# This creates a U-shaped curve
```

#### Exponential Function

```
f(x) = a^x
```

- Grows (or decays) very fast

```python
y = 2**x      # Grows as x increases
y = 0.5**x    # Shrinks as x increases (decay)
```

#### Logarithmic Function

```
f(x) = log(x)
```

- The inverse of exponential
- Grows slowly

```python
y = np.log(x)  # Natural logarithm
y = np.log10(x)  # Base 10 logarithm
```

### 2.3 Why These Functions Matter

| Function    | ML Use Case                               |
| ----------- | ----------------------------------------- |
| Linear      | Linear regression, neural network layers  |
| Quadratic   | Loss functions, regularization            |
| Exponential | Learning rate decay, activation functions |
| Logarithmic | Log-loss in classification, entropy       |

---

## 3. Linear Algebra

Linear algebra is the math of **vectors** and **matrices**. It's the foundation of deep learning!

### 3.1 Vectors

A **vector** is a list of numbers - think of it as a point in space or a direction.

```python
import numpy as np

# Creating vectors
v1 = np.array([1, 2, 3])      # 3D vector
v2 = np.array([4, 5, 6])

# Vector addition
result = v1 + v2  # [5, 7, 9]

# Scalar multiplication
doubled = 2 * v1  # [2, 4, 6]

# Dot product (how similar are two vectors?)
dot = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32

# Magnitude (length) of vector
mag = np.linalg.norm(v1)  # sqrt(1² + 2² + 3²) = sqrt(14)
```

**Visualizing Vectors:**

```
Vector [1, 2] → Point at x=1, y=2
        |
        |       * (1,2)
        |
        +------------→ x
```

### 3.2 Matrices

A **matrix** is a 2D array of numbers - like a spreadsheet or table.

```python
# Creating matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2 rows, 3 columns

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])  # 3 rows, 2 columns

# Matrix multiplication
C = np.dot(A, B)  # Result: 2x2 matrix
# Important: A's columns must match B's rows!

# Identity matrix (like 1 for numbers)
I = np.eye(3)  # 3x3 identity

# Transpose (swap rows and columns)
A_T = A.T
```

### 3.3 Matrix Operations in ML

**Linear Transformation:**

```python
# y = Wx + b
# This is exactly what happens in a neural network layer!

W = np.random.randn(10, 5)  # Weight matrix
x = np.random.randn(5)       # Input vector
b = np.random.randn(10)       # Bias vector

y = np.dot(W, x) + b  # Output
```

**Image as Matrix:**

```
A grayscale image is a matrix where each element is a pixel value (0-255)

[[255, 200, 150],
 [180, 120,  90],   → This IS the image!
 [100,  80,  60]]
```

### 3.4 Eigenvalues and Eigenvectors

This sounds scary but it's simple!

**Simple explanation:**

- An eigenvector doesn't change direction when a transformation is applied
- The eigenvalue tells you how much it stretched

```python
# In numpy
eigenvalues, eigenvectors = np.linalg.eig(A)

# This is used in:
# - PCA (Principal Component Analysis) for dimensionality reduction
# - Google's PageRank algorithm
# - Understanding neural network dynamics
```

### 3.5 NumPy Cheatsheet for Linear Algebra

```python
import numpy as np

# Create arrays
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4]])

# Shapes
vector.shape      # (3,)
matrix.shape      # (2, 2)

# Element-wise operations
matrix + 5        # Add to every element
matrix * 2        # Multiply every element

# Matrix operations
np.dot(A, B)      # Matrix multiplication
np.transpose(A)   # Transpose
np.linalg.inv(A)  # Inverse
np.linalg.det(A)  # Determinant
np.linalg.norm(A) # Norm (magnitude)
```

---

## 4. Calculus Basics

Calculus helps us understand **how things change** - essential for training AI models!

### 4.1 Derivatives

A **derivative** tells you the rate of change - like the speed in a car.

```python
# Simple derivatives:

# f(x) = x²
# f'(x) = 2x  (derivative)

# At x = 3:
# f(3) = 9
# f'(3) = 6  (slope at that point)
```

**Why derivatives in ML?**

- Derivatives tell us which direction to move to minimize error
- Used in **gradient descent** to train models

```python
# Numerical derivative
def derivative(f, x, h=0.0001):
    """Calculate derivative of f at point x"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Example: derivative of x² at x = 3
f = lambda x: x**2
print(derivative(f, 3))  # Approximately 6.0
```

### 4.2 Partial Derivatives

When a function has multiple inputs, we take derivatives with respect to each one:

```python
# f(x, y) = x² + y³
# ∂f/∂x = 2x  (treat y as constant)
# ∂f/∂y = 3y²  (treat x as constant)

# In ML, we have many parameters!
# ∂L/∂w₁, ∂L/∂w₂, ... (gradient of loss with respect to weights)
```

### 4.3 The Gradient

A **gradient** is a vector of all partial derivatives - it points in the direction of steepest increase.

```python
# Gradient of f(x, y) = x² + y²
# ∇f = [2x, 2y]

# At point (3, 4):
# ∇f = [6, 8]
# The gradient points in direction of steepest increase
# The negative gradient points to the minimum!
```

### 4.4 Chain Rule

The chain rule helps us differentiate composite functions - crucial for backpropagation in neural networks!

```python
# If y = f(g(x))
# Then dy/dx = f'(g(x)) * g'(x)

# Example: y = (x² + 1)³
# Let u = x² + 1
# y = u³
# dy/du = 3u²
# du/dx = 2x
# dy/dx = 3(x² + 1)² * 2x = 6x(x² + 1)²
```

**In Neural Networks:**

```
Loss = L(prediction, target)
prediction = f(weights)
weights = g(input)

dLoss/dweights = dLoss/dprediction × dprediction/dweights
```

This is exactly **backpropagation**!

### 4.5 Integrals

An **integral** is the opposite of a derivative - it finds the area under a curve.

```python
# Definite integral from a to b:
# ∫[a to b] f(x) dx

# Example: ∫[0 to 2] x² dx = [x³/3] from 0 to 2 = 8/3 - 0 = 8/3

# In probability:
# ∫ p(x) dx = 1  (probability distribution sums to 1)
```

---

## 5. Probability & Statistics

Probability helps us handle **uncertainty** - essential for ML!

### 5.1 Basic Probability

```python
# Probability is between 0 and 1
# P(event) = favorable outcomes / total outcomes

# Example: Rolling a 6 on a fair die
P_six = 1/6  # ≈ 0.167

# Complementary probability
P_not_six = 1 - P_six  # 5/6 ≈ 0.833
```

### 5.2 Important Rules

**Addition Rule:**

```
P(A or B) = P(A) + P(B) - P(A and B)
```

**Multiplication Rule (Independent Events):**

```
P(A and B) = P(A) × P(B)
```

**Conditional Probability:**

```
P(A | B) = P(A and B) / P(B)
```

"This is the probability of A given B has occurred"

**Bayes' Theorem:**

```
P(A | B) = P(B | A) × P(A) / P(B)
```

This is super important in ML! Used in:

- Naive Bayes classifiers
- Bayesian neural networks
- Many inference methods

```python
# Bayes' Theorem example:
# P(Disease | Test Positive) = ?

# P(Disease) = 0.01           (1% of people have disease)
# P(Positive | Disease) = 0.99 (test is 99% accurate for sick people)
# P(Positive | No Disease) = 0.05 (5% false positive rate)

P_disease = 0.01
P_positive_given_disease = 0.99
P_positive_given_no_disease = 0.05
P_no_disease = 0.99

# P(Positive) = P(Pos|D) * P(D) + P(Pos|~D) * P(~D)
P_positive = (P_positive_given_disease * P_disease +
              P_positive_given_no_disease * P_no_disease)

# Bayes: P(Disease | Positive) = P(Pos|D) * P(D) / P(Pos)
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive
print(P_disease_given_positive)  # ≈ 0.167 (16.7%)
```

### 5.3 Probability Distributions

A **distribution** describes how probabilities are spread across possible outcomes.

**Uniform Distribution:**

```python
# Every outcome equally likely
import numpy as np

rolls = np.random.randint(1, 7, 10000)  # Roll die 10000 times
# Each number (1-6) appears roughly 1667 times
```

**Normal (Gaussian) Distribution:**

```
The famous "bell curve"!
Most values cluster around the mean.
```

```python
import numpy as np

# Generate normally distributed data
data = np.random.normal(loc=0, scale=1, size=10000)
# loc = mean, scale = standard deviation

# 68% of data within 1 std
# 95% within 2 std
# 99.7% within 3 std
```

**Bernoulli Distribution:**

- Only two outcomes: success (p) or failure (1-p)
- Used for binary classification

```python
# Flip a biased coin
result = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% chance of 1
```

### 5.4 Statistics Basics

```python
import numpy as np
import statistics

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Mean (average)
mean = np.mean(data)           # 5.5
mean = statistics.mean(data)   # 5.5

# Median (middle value)
median = np.median(data)       # 5.5
median = statistics.median(data)

# Mode (most common)
mode = statistics.mode([1, 2, 2, 3, 3, 3])  # 3

# Variance (spread of data)
variance = np.var(data)        # 8.25

# Standard deviation
std = np.std(data)             # 2.87

# Percentiles
p25 = np.percentile(data, 25)  # 25th percentile
p50 = np.percentile(data, 50)  # 50th (median)
p75 = np.percentile(data, 75)  # 75th percentile
```

### 5.5 Correlation

**Correlation** measures how two variables move together:

```python
import numpy as np

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Pearson correlation coefficient
correlation = np.corrcoef(x, y)[0, 1]

# -1: Perfect negative correlation
#  0: No correlation
# +1: Perfect positive correlation
```

---

## 6. Optimization

Optimization is about **finding the best** - essential for training ML models!

### 6.1 What is Optimization?

In ML, we want to **minimize error** (or **maximize accuracy**).

```python
# Our prediction function
def predict(x, weights):
    return weights[0] * x + weights[1]

# Our error function (Mean Squared Error)
def mse(predictions, actual):
    return np.mean((predictions - actual) ** 2)

# Goal: Find weights that minimize MSE!
```

### 6.2 Gradient Descent

**Gradient descent** is the workhorse of ML optimization:

1. Start with random weights
2. Calculate the gradient (which way increases error)
3. Move in the opposite direction (reduce error)
4. Repeat until error stops decreasing

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    """Simple gradient descent for linear regression"""

    # Start with random weights
    m = 0  # slope
    b = 0  # intercept

    n = len(X)

    for _ in range(epochs):
        # Predictions
        y_pred = m * X + b

        # Calculate gradients
        dm = (-2/n) * np.sum(X * (y - y_pred))  # derivative wrt m
        db = (-2/n) * np.sum(y - y_pred)        # derivative wrt b

        # Update weights
        m = m - learning_rate * dm
        b = b - learning_rate * db

    return m, b

# Example
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

m, b = gradient_descent(X, y, learning_rate=0.01, epochs=1000)
print(f"Best line: y = {m:.2f}x + {b:.2f}")
```

### 6.3 Learning Rate

The **learning rate** controls how big steps we take:

```python
# Too small: Takes forever to converge
# Too big: Overshoots, may never find minimum
# Just right: Finds minimum efficiently!

# Visualizing learning rates:
"""
Learning Rate = 0.001 (too small)
    ↓          ↓          ↓          ↓
    ●----------●----------●----------●  (slow progress)

Learning Rate = 0.1 (good)
         ↓         ↓        ↓        ↓
              ●-----------●-----------●  (fast convergence)

Learning Rate = 1.5 (too big)
    ↓
           ↓                        (overshoots!)
               ↓
                     ↓
                         →  (never converges)
"""
```

### 6.4 Common Optimizers

```python
# In PyTorch or TensorFlow:

# SGD (Stochastic Gradient Descent)
# - Classic, but can be slow
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Adam (Adaptive Moment Estimation)
# - Most popular, usually works best!
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# RMSprop
# - Good for recurrent neural networks
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
```

### 6.5 Loss Functions

The **loss function** measures how wrong our predictions are:

| Loss Function            | Use Case                        |
| ------------------------ | ------------------------------- |
| Mean Squared Error (MSE) | Regression                      |
| Cross-Entropy            | Classification                  |
| Binary Cross-Entropy     | Binary Classification           |
| Huber Loss               | Regression (robust to outliers) |

```python
# MSE for regression
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Cross-Entropy for classification
def cross_entropy_loss(y_pred, y_true):
    # y_pred: probabilities, y_true: class labels
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

---

## 7. Additional Essential Topics

### 7.1 Information Theory

Information theory is crucial for **NLP, machine learning, and neural networks**. It helps us measure information and make decisions!

**Entropy** - measures uncertainty:

```python
import numpy as np

def entropy(probabilities):
    """Calculate Shannon entropy"""
    # Remove zero probabilities to avoid log(0)
    p = np.array(probabilities)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

# Example: Coin flip
p_heads = 0.5
p_tails = 0.5
print(f"Fair coin entropy: {entropy([p_heads, p_tails]):.2f} bits")

# Biased coin (less uncertainty = lower entropy)
p_heads = 0.9
p_tails = 0.1
print(f"Biased coin entropy: {entropy([p_heads, p_tails]):.2f} bits")
```

**Why it matters:**

- Cross-entropy loss in classification
- KL divergence for distribution matching
- Perplexity in language models
- Information gain for decision trees

**KL Divergence** - measures difference between two distributions:

```python
def kl_divergence(p, q):
    """KL(P || Q) - how different is Q from P"""
    p = np.array(p)
    q = np.array(q)
    # Add small epsilon to avoid log(0)
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

# P = [0.5, 0.5], Q = [0.9, 0.1]
print(f"KL divergence: {kl_divergence([0.5, 0.5], [0.9, 0.1]):.4f}")
```

### 7.2 Trigonometry for ML

Trigonometric functions appear in **Fourier transforms, signal processing, and some activation functions**:

```python
import numpy as np
import matplotlib.pyplot as plt

# Basic trig functions
x = np.linspace(0, 2 * np.pi, 100)

y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)

# Why in ML?
# - sin/cos used in positional encoding (Transformers)
# - Fourier transforms for convolution
# - Periodic features in time series
```

**Softmax as normalized exponential:**

```python
def softmax(x):
    """Convert logits to probabilities"""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

logits = [2.0, 1.0, 0.1]
print(f"Softmax: {softmax(logits)}")  # Sum to 1.0
```

### 7.3 Matrix Decomposition

Matrix decomposition makes computations **faster and more stable**:

**Singular Value Decomposition (SVD):**

```python
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])

# SVD: A = U * Σ * V^T
U, s, Vt = np.linalg.svd(A)

print(f"U shape: {U.shape}")
print(f"Singular values: {s}")
print(f"V^T shape: {Vt.shape}")

# Use cases:
# - Dimensionality reduction (like PCA)
# - Image compression
# - Recommender systems
# - Solving linear systems
```

**Eigenvalue Decomposition:**

```python
# For square symmetric matrices
A = np.array([[4, 2], [2, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Use cases:
# - PCA (Principal Component Analysis)
# - PageRank
# - Markov chains
# - Quantum computing
```

### 7.4 Lagrange Multipliers

Used for **constrained optimization** (like SVMs):

```python
# Problem: Maximize f(x,y) = x + y
# Subject to: x² + y² = 1 (on a circle)

# Lagrange multiplier method:
# L(x, y, λ) = f(x,y) - λ(g(x,y) - c)
# Set ∂L/∂x = ∂L/∂y = ∂L/∂λ = 0

# In SVM, this finds the maximum margin hyperplane!

# Simple example: minimize x² + y² subject to x + y = 1
# Solution: x = 0.5, y = 0.5 (point closest to origin on line)
```

### 7.5 Taylor Series

Used in **optimization and approximations**:

```python
# Approximate functions with polynomials

# e^x = 1 + x + x²/2! + x³/3! + ...
def taylor_exp(x, n=5):
    """Approximate e^x using n terms"""
    result = 0
    for i in range(n):
        result += x**i / np.math.factorial(i)
    return result

print(f"e^1 ≈ {taylor_exp(1, 10):.6f}")  # Close to 2.71828

# sin(x) = x - x³/3! + x⁵/5! - ...
# cos(x) = 1 - x²/2! + x⁴/4! - ...

# Used in:
# - Newton-Raphson optimization
# - Numerical methods
# - Function approximations
```

### 7.6 Covariance Matrices

Essential for **multivariate statistics and dimensionality reduction**:

```python
import numpy as np

# Generate 2D data
x = np.random.randn(100)
y = 2*x + np.random.randn(100) * 0.5  # Correlated with x

# Stack into matrix
data = np.stack([x, y], axis=1)

# Covariance matrix
cov_matrix = np.cov(data.T)

print("Covariance Matrix:")
print(cov_matrix)
# [[var(x), cov(x,y)],
#  [cov(x,y), var(y)]]

# Covariance tells us:
# - Positive: variables move together
# - Negative: move in opposite directions
# - Zero: no relationship
```

### 7.7 Advanced Optimization Concepts

**Momentum** - helps escape local minima:

```python
# Like a ball rolling down a hill, gaining momentum
def gradient_descent_momentum(X, y, momentum=0.9, lr=0.01, epochs=100):
    m = 0
    b = 0
    v_m = 0  # velocity
    v_b = 0

    n = len(X)

    for _ in range(epochs):
        y_pred = m * X + b
        dm = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)

        # Update with momentum
        v_m = momentum * v_m - lr * dm
        v_b = momentum * v_b - lr * db
        m = m + v_m
        b = b + v_b

    return m, b
```

**Adam Optimizer** (combines momentum + adaptive learning rates):

```python
# Adam = Adaptive Moment Estimation
# 1. Momentum (like a ball rolling)
# 2. Adaptive learning rates (different rates for each parameter)

# In practice, just use PyTorch or TensorFlow:
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 8. Summary & Next Steps

### What You've Learned

✅ **Basic Algebra** - Variables, equations, functions
✅ **Linear Algebra** - Vectors, matrices, operations
✅ **Calculus** - Derivatives, gradients, chain rule
✅ **Probability** - Distributions, Bayes' theorem
✅ **Statistics** - Mean, variance, correlation
✅ **Optimization** - Gradient descent, loss functions
✅ **Information Theory** - Entropy, KL divergence
✅ **Matrix Decomposition** - SVD, Eigenvalues
✅ **Advanced Topics** - Lagrange multipliers, Taylor series, Covariance

### Topics to Explore Next

1. **For Deep Learning:**
   - Matrix derivatives
   - Backpropagation algorithm
   - Jacobian and Hessian matrices

2. **For NLP:**
   - Information theory (entropy, perplexity)
   - Attention mechanisms

3. **For Reinforcement Learning:**
   - Markov Decision Processes
   - Bellman equations

### Recommended Resources

- **Khan Academy** - Great for foundational math
- **3Blue1Brown** - Amazing visual explanations
- **StatQuest** - Clear statistics explanations
- **3Blue1Brown's "Neural Networks" series** - Visual introduction

### Practice Projects

1. Implement gradient descent from scratch
2. Build a simple linear regression
3. Calculate probabilities for a Bayesian classifier
4. Visualize different activation functions
5. Implement a simple neural network (forward + backward pass)

---

## Quick Reference Cheatsheet

```python
# Essential imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Vector operations
v = np.array([1, 2, 3])
np.dot(v, v)           # Dot product
np.linalg.norm(v)     # Magnitude

# Matrix operations
A = np.array([[1, 2], [3, 4]])
np.dot(A, A)          # Matrix multiplication
A.T                   # Transpose
np.linalg.inv(A)      # Inverse

# Statistics
np.mean(data)
np.std(data)
np.corrcoef(x, y)

# Derivatives (numerical)
def derivative(f, x, h=0.001):
    return (f(x + h) - f(x - h)) / (2 * h)

# Gradient descent
def gradient_descent(grad_func, x, lr=0.01, epochs=100):
    for _ in range(epochs):
        x = x - lr * grad_func(x)
    return x
```

---

**Keep Learning!**

Mathematics for ML is a journey. Don't try to learn everything at once. Focus on one topic, practice with code, then move to the next. You'll be building AI models before you know it!

---

_Created for beginners wanting to learn AI/ML. Based on essential mathematical foundations needed for understanding machine learning algorithms._
