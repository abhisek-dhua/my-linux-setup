# Ultimate AI & ML Documentation: From Beginner to Advanced

---

## Table of Contents

1. [Introduction to Artificial Intelligence and Machine Learning](#1-introduction-to-artificial-intelligence-and-machine-learning)
2. [Python Programming Fundamentals](#2-python-programming-fundamentals)
3. [Mathematics for Machine Learning](#3-mathematics-for-machine-learning)
4. [Data Science Foundations](#4-data-science-foundations)
5. [Introduction to Machine Learning](#5-introduction-to-machine-learning)
6. [Supervised Learning Algorithms](#6-supervised-learning-algorithms)
7. [Unsupervised Learning Algorithms](#7-unsupervised-learning-algorithms)
8. [Model Evaluation and Validation](#8-model-evaluation-and-validation)
9. [Feature Engineering and Selection](#9-feature-engineering-and-selection)
10. [Deep Learning Fundamentals](#10-deep-learning-fundamentals)
11. [Neural Network Architectures](#11-neural-network-architectures)
12. [Natural Language Processing](#12-natural-language-processing)
13. [Computer Vision](#13-computer-vision)
14. [Reinforcement Learning](#14-reinforcement-learning)
15. [Advanced Deep Learning](#15-advanced-deep-learning)
16. [Model Optimization and Deployment](#16-model-optimization-and-deployment)
17. [MLOps and Production Systems](#17-mlops-and-production-systems)
18. [Specialized Topics and Emerging Technologies](#18-specialized-topics-and-emerging-technologies)
19. [Ethics, Bias, and Responsible AI](#19-ethics-bias-and-responsible-ai)
20. [Career Paths and Resources](#20-career-paths-and-resources)

---

# Part 1: Introduction & Beginner Level

---

## 1. Introduction to Artificial Intelligence and Machine Learning

### 1.1 What is Artificial Intelligence?

Artificial Intelligence (AI) is a broad field of computer science focused on creating systems capable of performing tasks that typically require human intelligence. These tasks include reasoning, learning, problem-solving, perception, language understanding, and decision-making.

AI encompasses various subfields and approaches:

- **Machine Learning (ML)**: Systems that learn from data without being explicitly programmed
- **Deep Learning (DL)**: Neural networks with multiple layers that learn hierarchical representations
- **Natural Language Processing (NLP)**: Enabling machines to understand and generate human language
- **Computer Vision**: Enabling machines to interpret and analyze visual information
- **Robotics**: Combining AI with physical systems and actuators
- **Expert Systems**: Rule-based systems that emulate human expertise in specific domains

### 1.2 A Brief History of AI

The history of AI spans over seven decades of research, development, and innovation:

**1940s-1950s: The Birth of AI**

- 1943: Warren McCulloch and Walter Pitts propose the first mathematical model of neural networks
- 1950: Alan Turing publishes "Computing Machinery and Intelligence," introducing the Turing Test
- 1956: The Dartmouth Conference marks the official birth of AI as a field
- 1957: Frank Rosenblatt develops the Perceptron, the first neural network

**1960s-1970s: Early Optimism and AI Winters**

- 1965: ELIZA, the first chatbot, is created by Joseph Weizenbaum
- 1966: Shakey the robot becomes the first mobile robot capable of reasoning
- 1969: Marvin Minsky and Seymour Papert publish "Perceptrons," showing limitations of single-layer networks
- 1974-1980: First "AI Winter" due to unmet expectations and funding cuts

**1980s: Expert Systems Rise**

- 1980: XCON (eXpert CONfigurer) becomes the first successful commercial expert system
- 1982: Japan's Fifth Generation Computer Project sparks international AI competition
- 1986: Backpropagation algorithm is popularized, enabling multi-layer neural networks

**1990s-2000s: Statistical Learning Era**

- 1997: IBM's Deep Blue defeats world chess champion Garry Kasparov
- 1998: Yann LeCun introduces LeNet-5 for handwritten digit recognition
- 2006: Geoffrey Hinton introduces "Deep Belief Networks," revitalizing deep learning
- 2011: IBM's Watson wins Jeopardy!

**2010s-Present: Deep Learning Revolution**

- 2012: AlexNet wins ImageNet competition, revolutionizing computer vision
- 2014: Generative Adversarial Networks (GANs) are introduced
- 2016: Google's AlphaGo defeats world champion Lee Sedol in Go
- 2020: GPT-3 demonstrates remarkable language generation capabilities
- 2022: ChatGPT brings conversational AI to mainstream
- 2023-2024: Large Language Models (LLMs) and multimodal AI systems advance rapidly

### 1.3 What is Machine Learning?

Machine Learning is a subset of AI that focuses on developing algorithms and statistical models that enable computers to perform tasks without explicit programming. Instead of following predefined rules, ML systems learn patterns from data.

**Key Characteristics of Machine Learning:**

1. **Data-Driven Learning**: ML algorithms improve performance by learning from data
2. **Pattern Recognition**: Identifying underlying patterns and relationships in data
3. **Generalization**: Applying learned patterns to new, unseen data
4. **Iterative Optimization**: Continuously improving through feedback and adjustment

### 1.4 Types of Machine Learning

Machine Learning is broadly categorized into three main types:

#### 1.4.1 Supervised Learning

In supervised learning, the algorithm learns from labeled data—data that includes both input features and the correct output (label). The goal is to learn a mapping function from inputs to outputs.

**Examples:**

- Email spam classification (input: email content, output: spam/not spam)
- House price prediction (input: features like size, location, output: price)
- Image classification (input: image, output: object category)

**Common Algorithms:**

- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- Neural Networks

**Use Cases:**

- Medical diagnosis
- Fraud detection
- Customer churn prediction
- Sentiment analysis

#### 1.4.2 Unsupervised Learning

In unsupervised learning, the algorithm learns from unlabeled data—data without predefined labels. The goal is to discover hidden patterns or structures within the data.

**Examples:**

- Customer segmentation (grouping similar customers)
- Anomaly detection (identifying unusual patterns)
- Dimensionality reduction (simplifying data while preserving structure)

**Common Algorithms:**

- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- Autoencoders
- t-SNE

**Use Cases:**

- Market basket analysis
- Document clustering
- Recommendation systems
- Data visualization

#### 1.4.3 Reinforcement Learning

In reinforcement learning, an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative rewards over time.

**Examples:**

- Game playing (Chess, Go, Atari games)
- Robotics control
- Autonomous driving
- Resource management

**Key Components:**

- Agent: The learner/decision maker
- Environment: The system the agent interacts with
- Action: The choices the agent can make
- Reward: Feedback from the environment
- Policy: The strategy the agent uses to select actions

**Common Algorithms:**

- Q-Learning
- Deep Q-Network (DQN)
- Policy Gradient methods
- Actor-Critic methods

### 1.5 The AI/ML/DL Relationship

Understanding the relationship between AI, ML, and Deep Learning:

```
┌─────────────────────────────────────────────┐
│           Artificial Intelligence           │
│  (Broadest concept - machines with          │
│   human-like intelligence)                  │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│           Machine Learning                  │
│  (AI subset - systems that learn from data) │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│           Deep Learning                     │
│  (ML subset - neural networks with          │
│   multiple layers)                          │
└─────────────────────────────────────────────┘
```

### 1.6 Why Machine Learning Matters

Machine Learning has become transformative due to several factors:

1. **Data Explosion**: The exponential growth of digital data provides rich training material
2. **Computing Power**: GPUs and specialized hardware enable faster training
3. **Algorithm Advances**: Improved algorithms and architectures yield better results
4. **Economic Value**: ML drives automation, prediction, and decision-making
5. **Accessibility**: Cloud platforms and open-source tools democratize ML

### 1.7 Common AI/ML Terminology

Understanding key terminology is essential:

- **Algorithm**: A step-by-step procedure for solving a problem
- **Model**: The learned representation from training data
- **Training**: The process of learning from data
- **Inference**: Using a trained model to make predictions
- **Feature**: An input variable used for prediction
- **Label**: The target output in supervised learning
- **Dataset**: A collection of data used for training/testing
- **Hyperparameter**: A parameter set before training
- **Overfitting**: Model learns training data too well, fails on new data
- **Underfitting**: Model is too simple to capture data patterns
- **Bias**: Systematic error in predictions
- **Variance**: Model's sensitivity to training data variations

### 1.8 The Machine Learning Workflow

A typical ML project follows these stages:

```
1. Problem Definition
   ├── Define the business problem
   ├── Identify the type of ML problem (classification, regression, etc.)
   └── Set success metrics

2. Data Collection
   ├── Gather relevant data
   ├── Ensure data quality and quantity
   └── Consider data privacy and ethics

3. Data Preparation
   ├── Clean and preprocess data
   ├── Handle missing values
   ├── Feature engineering
   └── Data augmentation

4. Model Selection
   ├── Choose appropriate algorithms
   ├── Consider model complexity
   └── Baseline models first

5. Training
   ├── Split data (train/validation/test)
   ├── Train the model
   ├── Tune hyperparameters
   └── Monitor training progress

6. Evaluation
   ├── Evaluate on validation set
   ├── Use appropriate metrics
   ├── Analyze errors
   └── Compare to baseline

7. Deployment
   ├── Export model
   ├── Set up inference pipeline
   ├── Monitor performance
   └── Plan for maintenance

8. Iteration
   ├── Gather feedback
   ├── Collect more data
   ├── Improve the model
   └── Repeat the process
```

### 1.9 Setting Up Your ML Environment

#### 1.9.1 Python Installation

Python is the dominant language for ML development. Install Python 3.8+ from python.org or use Anaconda:

```bash
# Using conda (recommended for ML)
conda create -n ml_env python=3.11
conda activate ml_env

# Or using virtualenv
python -m venv ml_env
source ml_env/bin/activate  # Linux/Mac
# or
ml_env\Scripts\activate     # Windows
```

#### 1.9.2 Essential Libraries

Install core ML libraries:

```bash
# Core data science libraries
pip install numpy pandas matplotlib seaborn

# Machine learning libraries
pip install scikit-learn

# Deep learning frameworks
pip install tensorflow        # Google
pip install torch             # PyTorch (Facebook)
# OR
pip install jax               # Google (advanced)

# Natural Language Processing
pip install nltk spacy transformers

# Computer Vision
pip install opencv-python pillow

# Utilities
pip install jupyterlab notebook
```

#### 1.9.3 Jupyter Notebooks

Jupyter notebooks are essential for ML development:

```bash
# Install JupyterLab (recommended)
pip install jupyterlab

# Or classic Jupyter Notebook
pip install notebook

# Start Jupyter
jupyter lab
# or
jupyter notebook
```

#### 1.9.4 Google Colab

Google Colab provides free GPU access in the cloud:

1. Go to colab.research.google.com
2. Create a new notebook
3. Enable GPU: Runtime → Change runtime type → GPU

### 1.10 Your First ML Project

Let's implement a simple ML project to classify iris flowers:

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the iris dataset
iris = load_iris()
X = iris.data          # Features: sepal length, sepal width, petal length, petal width
y = iris.target        # Labels: 0, 1, 2 (three species)

# Create a DataFrame for easier exploration
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nSpecies distribution:\n{df['species'].value_counts()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Make a prediction for a new flower
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example measurements
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
predicted_species = iris.target_names[prediction[0]]
print(f"\nPredicted species: {predicted_species}")
```

### 1.11 Introduction to Machine Learning Frameworks

#### 1.11.1 Scikit-learn

Scikit-learn is the most popular ML library for traditional ML algorithms:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Support Vector Machine
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
```

#### 1.11.2 TensorFlow

TensorFlow is Google's open-source deep learning framework:

```python
import tensorflow as tf
from tensorflow import keras

# Build a simple neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)
```

#### 1.11.3 PyTorch

PyTorch is Facebook's deep learning framework, known for dynamic computation:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(torch.FloatTensor(X_train))
    loss = criterion(outputs, torch.LongTensor(y_train))
    loss.backward()
    optimizer.step()
```

### 1.12 Best Practices for Beginners

#### 1.12.1 Coding Practices

1. **Start Simple**: Begin with baseline models before complex ones
2. **Understand Your Data**: Always explore and visualize your data first
3. **Document Your Work**: Comment your code and keep track of experiments
4. **Version Control**: Use Git for your projects
5. **Reproducibility**: Set random seeds for consistent results

```python
# Set random seeds for reproducibility
import numpy as np
import random
import torch

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
```

#### 1.12.2 Model Development Practices

1. **Always Split Data**: Never evaluate on training data
2. **Cross-Validation**: Use k-fold cross-validation for robust evaluation
3. **Feature Scaling**: Normalize or standardize features
4. **Avoid Data Leakage**: Ensure test data is truly unseen during training
5. **Hyperparameter Tuning**: Systematic search over parameter spaces

```python
from sklearn.model_selection import cross_val_score

# Cross-validation example
model = LogisticRegression(max_iter=200)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
```

#### 1.12.3 Debugging ML Models

Common issues and solutions:

```python
# Issue 1: Model not learning (loss not decreasing)
# Solutions:
# - Increase learning rate
# - Check for data preprocessing errors
# - Verify model architecture
# - Ensure sufficient training epochs

# Issue 2: Overfitting (high train accuracy, low test accuracy)
# Solutions:
# - Add regularization (L1/L2)
# - Use dropout in neural networks
# - Reduce model complexity
# - Increase training data
# - Use cross-validation

# Issue 3: Underfitting (low accuracy on both train and test)
# Solutions:
# - Increase model complexity
# - Add more features
# - Train longer
# - Reduce regularization

# Issue 4: Gradient explosion/vanishing
# Solutions:
# - Use gradient clipping
# - Proper weight initialization
# - Batch normalization
# - Use appropriate activation functions
```

### 1.13 Introduction to Data Preprocessing

Data preprocessing is crucial for ML success:

#### 1.13.1 Handling Missing Values

```python
import pandas as pd
import numpy as np

# Create sample data with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, 4, 5]
})

# Check for missing values
print(df.isnull().sum())

# Method 1: Remove rows with missing values
df_cleaned = df.dropna()

# Method 2: Fill with a specific value
df_filled = df.fillna(0)

# Method 3: Fill with mean/median
df['A'] = df['A'].fillna(df['A'].mean())
df['B'] = df['B'].fillna(df['B'].median())

# Method 4: Forward/backward fill
df_filled = df.fillna(method='ffill')  # Forward fill
df_filled = df.fillna(method='bfill')  # Backward fill
```

#### 1.13.2 Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding (for ordinal categories)
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])

# One-Hot Encoding (for nominal categories)
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')

# Using sklearn's OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)
X_encoded = ct.fit_transform(X)
```

#### 1.13.3 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (z-score normalization)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# MinMaxScaler (0 to 1 range)
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)

# RobustScaler (uses median, good for outliers)
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

### 1.14 Introduction to Data Visualization

Visualization is essential for understanding data:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Basic plot
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Basic Plot')
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot')
plt.colorbar()
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(X[:, 0], bins=30, edgecolor='black')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal length (cm)', data=df)
plt.title('Box Plot')
plt.show()

# Pair plot (seaborn)
sns.pairplot(df, hue='species')
plt.show()
```

### 1.15 Summary and Next Steps

In this chapter, you've learned:

1. **What AI and ML are** and how they relate to each other
2. **The history of AI** from its origins to modern deep learning
3. **Types of ML**: Supervised, Unsupervised, and Reinforcement Learning
4. **The ML workflow** from problem definition to deployment
5. **Setting up your ML environment** with Python and essential libraries
6. **Your first ML project** with the iris dataset
7. **Popular ML frameworks** like scikit-learn, TensorFlow, and PyTorch
8. **Best practices** for coding, model development, and debugging
9. **Data preprocessing** techniques
10. **Data visualization** basics

**Next Steps:**

- Practice with more datasets
- Learn Python in depth
- Study linear algebra and statistics
- Explore different ML algorithms

---

## 2. Python Programming Fundamentals

### 2.1 Why Python for ML?

Python has become the de facto language for machine learning due to:

1. **Simple Syntax**: Easy to learn and read
2. **Rich Ecosystem**: Extensive libraries for data science
3. **Community Support**: Large, active community
4. **Integration**: Works well with other languages and tools
5. **Production Readiness**: Used in production systems at major companies

### 2.2 Python Basics

#### 2.2.1 Variables and Data Types

```python
# Variables
name = "Machine Learning"  # String
version = 3.11             # Integer
price = 99.99              # Float
is_popular = True          # Boolean

# Multiple assignment
x, y, z = 1, 2, 3

# Type checking
print(type(name))  # <class 'str'>
print(type(price))  # <class 'float'>

# Type conversion
int_to_float = float(5)
float_to_int = int(5.7)
str_to_int = int("10")
```

#### 2.2.2 Collections

```python
# Lists (mutable, ordered)
numbers = [1, 2, 3, 4, 5]
numbers.append(6)
numbers[0] = 0
print(numbers[1:3])  # [2, 3]

# Tuples (immutable, ordered)
coordinates = (10, 20, 30)
print(coordinates[0])  # 10

# Dictionaries (key-value pairs)
person = {
    "name": "John",
    "age": 30,
    "city": "New York"
}
print(person["name"])  # John
person["job"] = "Engineer"

# Sets (unique, unordered)
unique_numbers = {1, 2, 3, 3, 4}  # {1, 2, 3, 4}
```

#### 2.2.3 Control Flow

```python
# If-else statements
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

# For loops
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

for item in ["apple", "banana", "cherry"]:
    print(item)

# While loops
count = 0
while count < 5:
    print(count)
    count += 1

# List comprehension
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### 2.3 Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Function with default parameters
def power(base, exponent=2):
    return base ** exponent

# Multiple return values
def divide(a, b):
    quotient = a // b
    remainder = a % b
    return quotient, remainder

q, r = divide(10, 3)

# Lambda functions (anonymous functions)
square = lambda x: x ** 2
add = lambda a, b: a + b

# Higher-order functions
def apply_operation(func, x):
    return func(x)

result = apply_operation(lambda x: x * 2, 5)  # 10

# Map, Filter, Reduce
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))  # [2, 4, 6, 8, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]
from functools import reduce
total = reduce(lambda a, b: a + b, numbers)  # 15
```

### 2.4 Object-Oriented Programming

```python
# Class definition
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def speak(self):
        return "Some sound"

    def __str__(self):
        return f"{self.name} is a {self.species}"

# Inheritance
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")
        self.breed = breed

    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

# Create instances
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", "Persian")

print(dog)  # Buddy is a Dog
print(dog.speak())  # Woof!
```

### 2.5 NumPy for Numerical Computing

NumPy is the foundation of numerical computing in Python:

```python
import numpy as np

# Creating arrays
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros(5)
ones = np.ones(5)
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Multi-dimensional arrays
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)  # (2, 3)

# Array operations
arr = np.array([1, 2, 3, 4, 5])
print(arr + 10)  # [11, 12, 13, 14, 15]
print(arr * 2)   # [2, 4, 6, 8, 10]
print(arr ** 2)  # [1, 4, 9, 16, 25]

# Element-wise operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)  # [5, 7, 9]
print(a * b)  # [4, 10, 18]

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)  # Matrix multiplication
print(A.dot(B))  # Alternative

# Array indexing and slicing
arr = np.array([0, 1, 2, 3, 4, 5])
print(arr[0])     # 0
print(arr[-1])    # 5
print(arr[1:4])   # [1, 2, 3]
print(arr[::2])   # [0, 2, 4]

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[0, 0])   # 1
print(matrix[0, :])   # [1, 2, 3]
print(matrix[:, 0])   # [1, 4, 7]
print(matrix[1:3, 1:3])  # [[5, 6], [8, 9]]

# Statistical functions
arr = np.array([1, 2, 3, 4, 5])
print(np.mean(arr))    # 3.0
print(np.median(arr))  # 3.0
print(np.std(arr))     # 1.414...
print(np.sum(arr))     # 15
print(np.min(arr))     # 1
print(np.max(arr))     # 5
```

### 2.6 Pandas for Data Manipulation

Pandas provides high-performance data structures:

```python
import pandas as pd

# Creating DataFrames
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data)

# Reading data
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')

# Exploring data
print(df.head())       # First 5 rows
print(df.tail())       # Last 5 rows
print(df.shape)        # Number of rows and columns
print(df.columns)      # Column names
print(df.dtypes)       # Data types
print(df.info())        # Summary info
print(df.describe())   # Statistical summary

# Selecting data
df['Name']           # Single column
df[['Name', 'Age']]  # Multiple columns
df.iloc[0]           # First row (by index)
df.loc[0]            # First row (by label)
df.iloc[0:3, 0:2]    # Slicing

# Filtering
df[df['Age'] > 30]              # Age > 30
df[(df['Age'] > 25) & (df['Salary'] > 60000)]

# Adding/modifying columns
df['Bonus'] = df['Salary'] * 0.1
df['Full Name'] = df['Name'] + ' Smith'

# Grouping and aggregation
df.groupby('Department')['Salary'].mean()
df.groupby(['Department', 'Year']).agg({
    'Salary': ['mean', 'sum'],
    'Age': 'max'
})

# Handling missing values
df.isnull()              # Check for missing
df.dropna()              # Remove rows with missing
df.fillna(0)             # Fill missing with value
df.fillna(df.mean())     # Fill with mean

# Merging DataFrames
pd.merge(df1, df2, on='key')
pd.concat([df1, df2])

# String operations
df['Name'].str.lower()
df['Name'].str.contains('John')
df['Email'].str.split('@')

# DateTime operations
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Pivot tables
df.pivot_table(values='Salary', index='Department', columns='Year', aggfunc='mean')
```

### 2.7 File Operations

```python
# Reading and writing text files
with open('file.txt', 'r') as f:
    content = f.read()

with open('file.txt', 'w') as f:
    f.write('Hello, World!')

# Reading and writing CSV files
import csv
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age'])
    writer.writerow(['Alice', 25])

# Using pandas for CSV
df.to_csv('output.csv', index=False)
df = pd.read_csv('input.csv')

# Working with JSON
import json
data = {'name': 'Alice', 'age': 25}
with open('data.json', 'w') as f:
    json.dump(data, f)

with open('data.json', 'r') as f:
    data = json.load(f)
```

### 2.8 Error Handling

```python
# Try-except blocks
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"Error: {e}")
else:
    print("No errors occurred")
finally:
    print("This always executes")

# Raising exceptions
def divide(a, b):
    if b == 0:
        raise ValueError("Divider cannot be zero")
    return a / b
```

### 2.9 Working with APIs

```python
import requests

# GET request
response = requests.get('https://api.example.com/data')
if response.status_code == 200:
    data = response.json()

# POST request
payload = {'name': 'Alice', 'age': 25}
response = requests.post('https://api.example.com/create', json=payload)

# Handling headers
headers = {'Authorization': 'Bearer token123'}
response = requests.get('https://api.example.com/protected', headers=headers)
```

### 2.10 Virtual Environments

Virtual environments isolate project dependencies:

```bash
# Create virtual environment
python -m venv myenv

# Activate (Linux/Mac)
source myenv/bin/activate

# Activate (Windows)
myenv\Scripts\activate

# Install packages
pip install numpy pandas

# Create requirements.txt
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt

# Deactivate
deactivate
```

---

## 3. Mathematics for Machine Learning

### 3.1 Why Math for ML?

Machine learning is built on mathematical foundations. Understanding the math behind algorithms helps you:

- Choose appropriate algorithms for your problem
- Debug and optimize models
- Understand limitations and assumptions
- Develop new approaches

### 3.2 Linear Algebra

#### 3.2.1 Vectors

A vector is a quantity with both magnitude and direction. In ML, vectors represent data points and features.

```python
import numpy as np

# Vectors in NumPy
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

# Vector addition
result = v + w  # [5, 7, 9]

# Scalar multiplication
result = 3 * v  # [3, 6, 9]

# Vector dot product
dot_product = np.dot(v, w)  # 1*4 + 2*5 + 3*6 = 32

# Vector magnitude (norm)
norm = np.linalg.norm(v)  # sqrt(1^2 + 2^2 + 3^2) = sqrt(14)

# Unit vector
unit = v / np.linalg.norm(v)

# Vector operations
# - Addition: element-wise
# - Subtraction: element-wise
# - Multiplication: element-wise (Hadamard product)
# - Division: element-wise
```

#### 3.2.2 Matrices

A matrix is a 2D array of numbers. In ML, matrices represent datasets where rows are samples and columns are features.

```python
# Creating matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

# Matrix shape
print(A.shape)  # (2, 3)

# Matrix addition
C = A + A  # Element-wise

# Matrix multiplication (dot product)
C = np.dot(A, B)  # Result: (2, 2)

# Transpose
A_T = A.T

# Identity matrix
I = np.eye(3)

# Matrix inverse
A_inv = np.linalg.inv(A)

# Determinant
det = np.linalg.det(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
```

#### 3.2.3 Matrix Operations in ML

```python
# Linear transformation
# y = Wx + b (common in neural networks)
W = np.random.randn(10, 5)  # Weight matrix
x = np.random.randn(5)      # Input vector
b = np.random.randn(10)    # Bias vector

y = np.dot(W, x) + b

# Batch processing
X = np.random.randn(32, 5)  # 32 samples, 5 features
Y = np.dot(X, W.T) + b     # Output: (32, 10)

# Cosine similarity (used in recommendations, NLP)
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

### 3.3 Calculus

#### 3.3.1 Derivatives

Derivatives measure how a function changes as its input changes. In ML, gradients guide optimization.

```python
# Numerical derivative
def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Derivative of x^2 at x=3
f = lambda x: x**2
print(derivative(f, 3))  # Approximately 6

# Partial derivatives
def f(x, y):
    return x**2 + y**2

def partial_x(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

def partial_y(f, x, y, h=1e-5):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)
```

#### 3.3.2 Gradient

A gradient is a vector of partial derivatives, pointing in the direction of steepest increase.

```python
# Gradient of a function
def gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Example: gradient of x^2 + y^2 at (3, 4)
f = lambda x: x[0]**2 + x[1]**2
grad = gradient(f, np.array([3.0, 4.0]))  # [6, 8]
```

#### 3.3.3 Chain Rule

The chain rule is essential for backpropagation in neural networks.

```python
# Chain rule: d/dx [f(g(x))] = f'(g(x)) * g'(x)

# Example: f(g(x)) where f(u) = u^2 and g(x) = 3x + 1
# f'(u) = 2u, g'(x) = 3
# d/dx = 2(3x + 1) * 3 = 6(3x + 1) = 18x + 6

# In neural networks:
# loss = L(a, y), a = f(z), z = Wx + b
# ∂L/∂W = ∂L/∂a * ∂a/∂z * ∂z/∂W
```

### 3.4 Probability and Statistics

#### 3.4.1 Basic Probability

```python
import numpy as np
import random

# Basic probability concepts
# P(A) = favorable outcomes / total outcomes

# Example: Rolling a 6 on a fair die
P_six = 1/6  # 0.167

# Independent events: P(A and B) = P(A) * P(B)
# P(two sixes) = (1/6) * (1/6) = 1/36

# Conditional probability: P(A|B) = P(A and B) / P(B)

# Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
```

#### 3.4.2 Probability Distributions

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Uniform distribution
uniform = np.random.uniform(0, 1, 1000)

# Normal (Gaussian) distribution
normal = np.random.normal(0, 1, 1000)  # mean=0, std=1

# Binomial distribution (discrete)
binomial = np.random.binomial(10, 0.5, 1000)  # n=10, p=0.5

# Poisson distribution
poisson = np.random.poisson(5, 1000)  # lambda=5

# Probability density functions
x = np.linspace(-4, 4, 100)

# Normal PDF
normal_pdf = stats.norm.pdf(x, 0, 1)

# Plot distributions
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(normal, bins=30, density=True, alpha=0.7)
plt.title('Normal Distribution')

plt.subplot(1, 3, 2)
plt.hist(binomial, bins=30, density=True, alpha=0.7)
plt.title('Binomial Distribution')

plt.subplot(1, 3, 3)
plt.hist(poisson, bins=30, density=True, alpha=0.7)
plt.title('Poisson Distribution')
plt.tight_layout()
plt.show()
```

#### 3.4.3 Statistical Measures

```python
import numpy as np
import pandas as pd

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Central tendency
mean = np.mean(data)           # Average
median = np.median(data)       # Middle value
mode = pd.Series(data).mode() # Most frequent

# Spread
variance = np.var(data)       # Average squared deviation
std = np.std(data)            # Square root of variance
range_val = max(data) - min(data)

# Quartiles
q1 = np.percentile(data, 25)   # 25th percentile
q2 = np.percentile(data, 50)   # 50th percentile (median)
q3 = np.percentile(data, 75)   # 75th percentile
iqr = q3 - q1                  # Interquartile range

# Skewness and Kurtosis
from scipy import stats
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

# Correlation
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
correlation = np.corrcoef(x, y)[0, 1]  # Pearson correlation
```

#### 3.4.4 Hypothesis Testing

```python
from scipy import stats
import numpy as np

# Sample data
group1 = [85, 87, 82, 86, 88, 85, 89]
group2 = [78, 82, 79, 81, 80, 77, 83]

# T-test (comparing means of two groups)
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Paired t-test (for paired samples)
t_stat, p_value = stats.ttest_rel(group1, group2)

# ANOVA (comparing means of 3+ groups)
f_stat, p_value = stats.f_oneway(group1, group2, [80, 82, 79])

# Chi-square test (for categorical data)
observed = [[10, 20], [30, 40]]
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

# Significance level
alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis (statistically significant)")
else:
    print("Fail to reject null hypothesis")
```

### 3.5 Optimization

#### 3.5.1 Gradient Descent

Gradient descent is the foundation of ML optimization:

```python
import numpy as np

def gradient_descent(grad_func, initial_x, learning_rate=0.1, n_iterations=100):
    """
    Basic gradient descent implementation.

    Args:
        grad_func: Function that computes gradient
        initial_x: Starting point
        learning_rate: Step size
        n_iterations: Number of iterations

    Returns:
        Optimal x value and history of values
    """
    x = initial_x.copy()
    history = [x.copy()]

    for i in range(n_iterations):
        gradient = grad_func(x)
        x = x - learning_rate * gradient
        history.append(x.copy())

    return x, history

# Example: Minimize f(x) = x^2
# Gradient: f'(x) = 2x
f = lambda x: x**2
grad_f = lambda x: 2 * x

optimal_x, history = gradient_descent(grad_f, np.array([5.0]),
                                      learning_rate=0.1, n_iterations=50)
print(f"Optimal x: {optimal_x}")
print(f"Minimum value: {f(optimal_x)}")
```

#### 3.5.2 Variants of Gradient Descent

```python
# Batch Gradient Descent (uses all data)
for epoch in range(n_epochs):
    gradients = compute_gradient_over_all_data()
    weights = weights - learning_rate * gradients

# Stochastic Gradient Descent (uses one sample)
for epoch in range(n_epochs):
    for sample in training_data:
        gradient = compute_gradient_for_sample(sample)
        weights = weights - learning_rate * gradient

# Mini-batch Gradient Descent (uses subset of data)
for epoch in range(n_epochs):
    for batch in create_batches(training_data, batch_size=32):
        gradients = compute_gradient_for_batch(batch)
        weights = weights - learning_rate * gradients
```

#### 3.5.3 Learning Rate Schedules

```python
# Step decay
def step_decay(epoch):
    initial_lr = 0.1
    drop = 0.5
    epochs_drop = 10
    return initial_lr * np.power(drop, np.floor(epoch / epochs_drop))

# Exponential decay
def exponential_decay(epoch):
    initial_lr = 0.1
    decay_rate = 0.95
    return initial_lr * np.power(decay_rate, epoch)

# Cosine annealing
def cosine_annealing(epoch, T_max=50):
    return 0.5 * (1 + np.cos(np.pi * epoch / T_max))

# Warmup
def warmup_schedule(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return 0.1 * (epoch + 1) / warmup_epochs
    return 0.1
```

### 3.6 Information Theory

```python
import numpy as np
from scipy import stats

# Entropy (uncertainty/information content)
def entropy(probabilities):
    """Calculate Shannon entropy."""
    # Filter out zero probabilities to avoid log(0)
    p = np.array(probabilities)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

# Example: Entropy of a fair coin flip
probabilities = [0.5, 0.5]
print(f"Entropy: {entropy(probabilities):.4f} bits")  # 1.0 bit

# Entropy of biased coin
probabilities = [0.9, 0.1]
print(f"Entropy: {entropy(probabilities):.4f} bits")  # ~0.47 bits

# Cross-entropy
def cross_entropy(p, q):
    """Calculate cross-entropy between two distributions."""
    p = np.array(p)
    q = np.array(q)
    return -np.sum(p * np.log2(q + 1e-10))

# KL Divergence (relative entropy)
def kl_divergence(p, q):
    """Calculate KL divergence: D(P || Q)"""
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log2(p / (q + 1e-10)))
```

---

## 4. Data Science Foundations

### 4.1 The Data Science Pipeline

Data science transforms raw data into actionable insights:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA SCIENCE PIPELINE                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1. Define Problem                                                    │
│     └── What question are we trying to answer?                        │
│                                                                        │
│  2. Collect Data                                                      │
│     └── APIs, Databases, Web Scraping, IoT                            │
│                                                                        │
│  3. Clean & Prepare Data                                              │
│     └── Missing values, outliers, transformations                    │
│                                                                        │
│  4. Explore & Visualize                                               │
│     └── Patterns, correlations, distributions                         │
│                                                                        │
│  5. Feature Engineering                                               │
│     └── Create meaningful features                                    │
│                                                                        │
│  6. Build Models                                                      │
│     └── Train, evaluate, select best model                            │
│                                                                        │
│  7. Communicate Results                                               │
│     └── Reports, dashboards, presentations                            │
│                                                                        │
│  8. Deploy & Monitor                                                  │
│     └── Production systems, monitoring                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Types of Data

#### 4.2.1 Structured Data

Organized in rows and columns (tables):

```python
import pandas as pd

# Tabular data
df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000],
    'is_active': [True, False, True]
})
```

**Data Types:**

- Numerical: integers, floats
- Categorical: nominal, ordinal
- Boolean: True/False
- DateTime: timestamps

#### 4.2.2 Unstructured Data

No predefined format (text, images, audio):

```python
# Text data
text = "This is a sample text document for NLP processing."

# Image data
from PIL import Image
img = Image.open('photo.jpg')
img_array = np.array(img)

# Audio data
import librosa
audio, sr = librosa.load('audio.wav')

# Video data
import cv2
video = cv2.VideoCapture('video.mp4')
```

#### 4.2.3 Semi-structured Data

Hierarchical structure (JSON, XML):

```python
import json

# JSON data
data = {
    "users": [
        {"name": "Alice", "age": 25, "skills": ["Python", "ML"]},
        {"name": "Bob", "age": 30, "skills": ["Java", "Web"]}
    ]
}

# Parse JSON
with open('data.json', 'r') as f:
    data = json.load(f)
```

### 4.3 Data Exploration

#### 4.3.1 Loading Data

```python
import pandas as pd

# CSV files
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', nrows=1000)  # First 1000 rows
df = pd.read_csv('data.csv', usecols=['col1', 'col2'])  # Specific columns

# Excel files
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('data.json')

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table_name', conn)

# Parquet (columnar format, efficient)
df = pd.read_parquet('data.parquet')

# Pickle
df = pd.read_pickle('data.pkl')
```

#### 4.3.2 Exploratory Data Analysis (EDA)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample dataset
df = pd.read_csv('titanic.csv')

# Basic information
print(df.shape)           # Dimensions
print(df.head())          # First rows
print(df.info())          # Data types and null counts
print(df.describe())      # Statistical summary

# Distribution analysis
print(df['Age'].describe())
print(df['Age'].value_counts())

# Missing values
print(df.isnull().sum())
print(df.isnull().sum() / len(df) * 100)  # Percentage

# Unique values
print(df['Pclass'].nunique())  # Number of unique values
print(df['Pclass'].unique())   # List of unique values
```

#### 4.3.3 Data Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution plots
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df['Age'].dropna(), bins=30, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age by Survival')

plt.subplot(1, 3, 3)
sns.violinplot(x='Pclass', y='Fare', data=df)
plt.title('Fare by Class')

plt.tight_layout()
plt.show()

# Categorical plots
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df['Survived'].value_counts().plot(kind='bar')
plt.title('Survival Count')

plt.subplot(1, 3, 2)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Sex')

plt.subplot(1, 3, 3)
sns.heatmap(pd.crosstab(df['Sex'], df['Survived']), annot=True, fmt='d')
plt.title('Survival Heatmap')

plt.tight_layout()
plt.show()

# Scatter plots
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, alpha=0.5)
plt.title('Age vs Fare')
plt.show()
```

### 4.4 Data Cleaning

#### 4.4.1 Handling Missing Data

```python
import pandas as pd
import numpy as np

# Check missing values
print(df.isnull().sum())

# Drop rows with missing values
df_clean = df.dropna()

# Drop columns with too many missing values
threshold = len(df) * 0.5
df_clean = df.dropna(thresh=threshold, axis=1)

# Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)  # For skewed data
df['Name'].fillna('Unknown', inplace=True)

# Forward/backward fill
df['Value'].fillna(method='ffill', inplace=True)
df['Value'].fillna(method='bfill', inplace=True)

# Interpolate (for time series)
df['Value'].interpolate(method='linear', inplace=True)

# Using sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

#### 4.4.2 Handling Outliers

```python
import numpy as np
import pandas as pd

# Detect outliers using IQR
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outliers
df_clean = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]

# Winsorization (cap outliers)
from scipy import stats
df['Age'] = stats.mstats.winsorize(df['Age'], limits=[0.05, 0.05])

# Z-score method
z_scores = np.abs(stats.zscore(df['Age']))
df_clean = df[z_scores < 3]
```

#### 4.4.3 Data Transformation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Log transformation (for skewed data)
df['Log_Fare'] = np.log1p(df['Fare'])

# Square root transformation
df['Sqrt_Fare'] = np.sqrt(df['Fare'])

# Box-Cox transformation (requires positive values)
from scipy import stats
df['BoxCox_Fare'], lambda_param = stats.boxcox(df['Fare'] + 1)

# Standardization (z-score)
scaler = StandardScaler()
df['Standardized_Age'] = scaler.fit_transform(df[['Age']])

# Normalization (0-1 range)
scaler = MinMaxScaler()
df['Normalized_Age'] = scaler.fit_transform(df[['Age']])

# Robust scaling (using median and IQR)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df['Robust_Age'] = scaler.fit_transform(df[['Age']])
```

### 4.5 Feature Engineering

#### 4.5.1 Creating New Features

```python
import pandas as pd
from datetime import datetime

# DateTime features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# String features
df['Name_Length'] = df['Name'].str.len()
df['Has_Email'] = df['Email'].notna().astype(int)

# Binning numerical features
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 100],
                          labels=['Child', 'Young', 'Middle', 'Senior'])

# Interaction features
df['Age_Salary_Ratio'] = df['Age'] / df['Salary']
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
```

#### 4.5.2 Encoding Categorical Variables

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label encoding
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])

# One-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# Using sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), ['Embarked'])],
    remainder='passthrough'
)

# Target encoding (for high cardinality)
target_mean = df.groupby('City')['Target'].mean()
df['City_Target'] = df['City'].map(target_mean)
```

#### 4.5.3 Feature Selection

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Univariate selection
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Feature importance from tree-based models
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).plot(kind='bar')

# Correlation analysis
corr_matrix = df.corr()
high_corr = corr_matrix[abs(corr_matrix['target']) > 0.3].index
```

### 4.6 Data Pipeline

#### 4.6.1 Building Pipelines with sklearn

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Define preprocessing steps
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Embarked', 'Pclass']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Use pipeline
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

## 5. Introduction to Machine Learning

### 5.1 What is Machine Learning?

Machine Learning is a method of teaching computers to learn from data without being explicitly programmed. Instead of writing explicit rules, we provide examples and let the computer discover patterns.

**Key Concepts:**

- **Learning from Data**: ML algorithms find patterns in data
- **Generalization**: The ability to apply learning to new data
- **Optimization**: Automatically improving performance
- **Representation**: How data is represented for learning

### 5.2 Types of ML Problems

#### 5.2.1 Regression (Continuous Output)

Predicting a continuous numerical value:

```python
# Examples:
# - House price prediction ($250,000)
# - Temperature forecasting (72.5°F)
# - Sales forecasting ($50,000/month)

from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data: Square footage -> House price
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200000, 300000, 400000, 500000, 600000])

model = LinearRegression()
model.fit(X, y)

# Predict price for 1800 sq ft
prediction = model.predict([[1800]])
print(f"Predicted price: ${prediction[0]:,.0f}")
```

#### 5.2.2 Classification (Discrete Output)

Predicting a category or class:

```python
# Examples:
# - Email spam detection (spam/not spam)
# - Disease diagnosis (has disease/doesn't have)
# - Image classification (cat/dog/bird)

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1)

model = LogisticRegression()
model.fit(X, y)

# Predict class for new sample
new_sample = [[0.5, 0.5]]
prediction = model.predict(new_sample)
print(f"Predicted class: {prediction[0]}")
```

#### 5.2.3 Clustering (Unsupervised Grouping)

Finding natural groups in data:

```python
# Examples:
# - Customer segmentation
# - Document clustering
# - Anomaly detection

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

print(f"Cluster centers: {kmeans.cluster_centers_}")
print(f"Number of samples in each cluster: {np.bincount(labels)}")
```

#### 5.2.4 Dimensionality Reduction

Reducing the number of features:

```python
# Examples:
# - Visualization (2D/3D projection)
# - Compression
# - Noise reduction

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
```

### 5.3 The ML Workflow

#### 5.3.1 Problem Definition

Before building any model, clearly define:

1. **What are we predicting?**
2. **What type of problem is it?**
3. **How will success be measured?**
4. **What data is available?**

```python
# Example: Customer Churn Prediction
# Goal: Predict which customers will cancel their subscription
# Type: Binary Classification (churn/not churn)
# Metric: Accuracy, Precision, Recall, F1, AUC-ROC
# Data: Customer demographics, usage patterns, billing history
```

#### 5.3.2 Data Preparation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('customer_data.csv')

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['income'].fillna(df['income'].mean(), inplace=True)

# Encode categorical variables
df['gender'] = df['gender'].map({'M': 0, 'F': 1})

# Select features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
```

#### 5.3.3 Model Selection

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Define models to try
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate each
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    results[name] = {'train': train_score, 'test': test_score}
    print(f"{name}: Train={train_score:.3f}, Test={test_score:.3f}")
```

#### 5.3.4 Model Training

```python
# Basic training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Training with cross-validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100),
    X_train, y_train, cv=5, scoring='accuracy'
)

print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
```

#### 5.3.5 Model Evaluation

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Classification Report
print(f"\n{classification_report(y_test, y_pred)}")
```

### 5.4 Underfitting and Overfitting

#### 5.4.1 Understanding the Problem

```
                    Model Complexity
                           │
    High Variance    │     High Bias
    (Overfitting)   │   (Underfitting)
                    │
         ───────────┼──────────────
                    │
              Optimal
            Complexity
                    │
                    └───────────────────
                           Error
                    Train Error ───────
                    Test Error  ───────
```

#### 5.4.2 Identifying the Problem

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Generate sample data
np.random.seed(42)
X = np.sort(np.random.rand(100) * 10)
y = np.sin(X) + np.random.randn(100) * 0.2

# Try different model complexities
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

degrees = [1, 3, 10, 15]
train_scores = []
test_scores = []

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('lr', LinearRegression())
    ])

    model.fit(X.reshape(-1, 1), y)
    train_scores.append(model.score(X.reshape(-1, 1), y))

    # Simulate test score (would use separate test set in practice)
    test_scores.append(model.score(X.reshape(-1, 1), y) - 0.1 * degree)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_scores, 'b-o', label='Train')
plt.plot(degrees, test_scores, 'r-o', label='Test')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Score')
plt.title('Underfitting vs Overfitting')
plt.legend()
plt.show()
```

#### 5.4.3 Solutions

```python
# Underfitting Solutions:
# - Increase model complexity
# - Add more features
# - Reduce regularization
# - Train longer

# Overfitting Solutions:
# - Reduce model complexity
# - More training data
# - Regularization (L1/L2)
# - Dropout (for neural networks)
# - Cross-validation

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier

# Regularization examples
ridge = Ridge(alpha=1.0)      # L2 regularization
lasso = Lasso(alpha=1.0)       # L1 regularization (sparsity)

# Random Forest (less prone to overfitting)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,         # Limit tree depth
    min_samples_split=5,  # Require more samples to split
    min_samples_leaf=2,    # Require more samples in leaves
    random_state=42
)
```

### 5.5 Bias-Variance Tradeoff

#### 5.5.1 Conceptual Understanding

- **Bias**: Error from overly simplistic assumptions
- **Variance**: Error from excessive model complexity

```
Total Error = Bias² + Variance + Irreducible Error

High Bias:   Underfitting (misses patterns)
High Variance: Overfitting (too sensitive to noise)
```

#### 5.5.2 Visualizing Bias-Variance

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Generate data with noise
np.random.seed(42)
X = np.linspace(0, 1, 20)
y = np.sin(2 * np.pi * X) + np.random.randn(20) * 0.3

# Test different polynomial degrees
degrees = range(1, 15)
train_scores = []
val_scores = []

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('lr', LinearRegression())
    ])

    # Training score
    model.fit(X.reshape(-1, 1), y)
    train_scores.append(model.score(X.reshape(-1, 1), y))

    # Cross-validation score (as proxy for test score)
    scores = cross_val_score(model, X.reshape(-1, 1), y, cv=5)
    val_scores.append(scores.mean())

# Plot
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_scores, 'b-o', label='Training Score')
plt.plot(degrees, val_scores, 'r-o', label='Cross-Validation Score')
plt.axvline(x=4, color='g', linestyle='--', label='Optimal Complexity')
plt.xlabel('Polynomial Degree')
plt.ylabel('Score')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True)
plt.show()
```

### 5.6 Regularization

Regularization adds constraints to prevent overfitting:

#### 5.6.1 L1 Regularization (Lasso)

```python
from sklearn.linear_model import Lasso
import numpy as np

# Lasso adds penalty: α * |weights|
# Tends to produce sparse solutions (some weights become zero)

X = np.random.randn(100, 10)
y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1

# Different alpha values
alphas = [0.001, 0.01, 0.1, 1.0]
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    print(f"Alpha: {alpha}, Non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
```

#### 5.6.2 L2 Regularization (Ridge)

```python
from sklearn.linear_model import Ridge

# Ridge adds penalty: α * weights²
# Shrinks weights towards zero but rarely makes them exactly zero

ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print(f"Ridge coefficients: {ridge.coef_}")
```

#### 5.6.3 Elastic Net (L1 + L2)

```python
from sklearn.linear_model import ElasticNet

# Combines L1 and L2 penalties
# Useful when there are correlated features

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio: 0=Ridge, 1=Lasso
elastic.fit(X, y)
```

### 5.7 Cross-Validation

#### 5.7.1 K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Basic K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")
```

#### 5.7.2 Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

# Maintains class distribution in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print(f"Stratified CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

#### 5.7.3 Leave-One-Out Cross-Validation

```python
from sklearn.model_selection import LeaveOneOut

# Use when data is very limited
loo = LeaveOneOut()

# Note: Can be computationally expensive
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

---

## 6. Supervised Learning Algorithms

### 6.1 Linear Regression

Linear regression is the foundation of supervised learning:

#### 6.1.1 Simple Linear Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: Years of experience vs Salary
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Model parameters
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R² Score: {model.score(X, y):.4f}")

# Prediction
years_of_exp = 7
predicted_salary = model.predict([[years_of_exp]])
print(f"Predicted salary for {years_of_exp} years: ${predicted_salary[0]:,.0f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
```

#### 6.1.2 Multiple Linear Regression

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Multiple features
df = pd.DataFrame({
    'sqft': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'bathrooms': [1, 2, 2, 3, 3],
    'age': [10, 5, 8, 3, 2],
    'price': [200000, 300000, 400000, 500000, 600000]
})

X = df[['sqft', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R² Score: {model.score(X, y):.4f}")
```

#### 6.1.3 Gradient Descent for Linear Regression

```python
import numpy as np

def gradient_descent_linear_regression(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Implement linear regression using gradient descent.
    """
    m, n = X.shape

    # Initialize parameters
    weights = np.zeros(n)
    bias = 0

    for _ in range(n_iterations):
        # Predictions
        y_pred = np.dot(X, weights) + bias

        # Calculate gradients
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

# Example usage
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Add bias term
X_b = np.hstack([np.ones((X.shape[0], 1)), X])

weights, bias = gradient_descent_linear_regression(X_b, y, learning_rate=0.1, n_iterations=1000)
print(f"Slope: {weights[1]:.4f}, Intercept: {bias:.4f}")
```

### 6.2 Logistic Regression

#### 6.2.1 Binary Classification

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Predicted classes: {np.unique(y_pred)}")
print(f"Probability of class 1: {y_prob[:5]}")
```

#### 6.2.2 Multinomial Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model with multinomial loss
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X, y)

# Predict
new_sample = [[5.1, 3.5, 1.4, 0.2]]
predicted_class = model.predict(new_sample)
predicted_prob = model.predict_proba(new_sample)

print(f"Predicted class: {iris.target_names[predicted_class[0]]}")
print(f"Probabilities: {dict(zip(iris.target_names, predicted_prob[0]))}")
```

### 6.3 Decision Trees

#### 6.3.1 Classification Trees

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate non-linear data
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Train tree with different depths
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for depth, ax in zip([1, 3, 10], axes):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X, y)

    accuracy = tree.score(X, y)
    ax.set_title(f"Depth={depth}, Accuracy={accuracy:.3f}")

    # Plot decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)

plt.tight_layout()
plt.show()
```

#### 6.3.2 Regression Trees

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.sort(np.random.rand(100) * 10)
y = np.sin(X) + np.random.randn(100) * 0.1

# Train regression tree
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X.reshape(-1, 1), y)

# Predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = tree.predict(X_test)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()
```

### 6.4 Random Forests

#### 6.4.1 Classification with Random Forests

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

rf.fit(X_train, y_train)

# Evaluate
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': range(20),
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))
```

#### 6.4.2 Regression with Random Forests

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.rand(100, 5)
y = (X[:, 0] + X[:, 1] * 2 + X[:, 2] ** 2 +
     np.random.randn(100) * 0.1)

# Train
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Score
r2 = rf.score(X, y)
print(f"R² Score: {r2:.4f}")

# Predictions
new_data = np.random.rand(5).reshape(1, -1)
prediction = rf.predict(new_data)
print(f"Prediction: {prediction[0]:.4f}")
```

### 6.5 Support Vector Machines

#### 6.5.1 SVM Classification

```python
from sklearn.svm import SVC
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np

# Generate circular data
X, y = make_circles(n_samples=200, factor=0.3, noise=0.1, random_state=42)

# Train SVM with different kernels
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

kernels = ['linear', 'rbf', 'poly']
for kernel, ax in zip(kernels, axes):
    svm = SVC(kernel=kernel, C=1.0, random_state=42)
    svm.fit(X, y)

    accuracy = svm.score(X, y)
    ax.set_title(f"Kernel: {kernel}, Accuracy: {accuracy:.3f}")

    # Plot decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)

plt.tight_layout()
plt.show()
```

#### 6.5.2 SVM Regression

```python
from sklearn.svm import SVR
import numpy as np

# Generate data
np.random.seed(42)
X = np.sort(np.random.rand(100) * 10)
y = np.sin(X) + np.random.randn(100) * 0.1

# Train SVR
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X.reshape(-1, 1), y)

# Predict
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = svr.predict(X_test)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Actual')
plt.plot(X_test, y_pred, color='red', label='SVR Prediction')
plt.legend()
plt.title('SVR Regression')
plt.show()
```

### 6.6 k-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Classification
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                            n_informative=2, n_clusters_per_class=1, random_state=42)

# Try different k values
k_values = [1, 3, 7, 15]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for k, ax in zip(k_values, axes.flatten()):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    accuracy = knn.score(X, y)
    ax.set_title(f"k={k}, Accuracy: {accuracy:.3f}")

    # Plot decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)

plt.tight_layout()
plt.show()
```

### 6.7 Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Gaussian Naive Bayes (continuous features)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"Gaussian NB Accuracy: {gnb.score(X_test, y_test):.3f}")

# Multinomial Naive Bayes (counts/categories)
# Example: Text classification with bag-of-words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["great movie", "bad film", "excellent acting", "terrible", "love it"]
labels = [1, 0, 1, 0, 1]

vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(texts)

mnb = MultinomialNB()
mnb.fit(X_text, labels)

# Predict
test_text = ["awesome film"]
X_test_text = vectorizer.transform(test_text)
prediction = mnb.predict(X_test_text)
print(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

### 6.8 Gradient Boosting

#### 6.8.1 XGBoost

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

# Feature importance
importance = model.feature_importances_
print(f"Top 5 important features: {np.argsort(importance)[-5:]}")
```

#### 6.8.2 LightGBM

```python
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"LightGBM Accuracy: {accuracy:.3f}")
```

---

## 7. Unsupervised Learning Algorithms

### 7.1 Clustering

#### 7.1.1 K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# Find optimal number of clusters using elbow method
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

# Train with optimal k
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', s=200, marker='X')
plt.title('K-Means Clustering')
plt.tight_layout()
plt.show()

print(f"Cluster Centers:\n{kmeans.cluster_centers_}")
```

#### 7.1.2 Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_moons

# Generate data
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Linkage matrix
Z = linkage(X, method='ward')

# Dendrogram
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
dendrogram(Z, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')

# Agglomerative clustering
hc = AgglomerativeClustering(n_clusters=2, linkage='ward')
y_pred = hc.fit_predict(X)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.title('Agglomerative Clustering')
plt.tight_layout()
plt.show()
```

#### 7.1.3 DBSCAN (Density-Based)

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate moon-shaped data
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_pred = dbscan.fit_predict(X)

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
plt.title('Original Data')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
# Highlight outliers
outliers = X[y_pred == -1]
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', marker='x', s=100, label='Outliers')
plt.title('DBSCAN Clustering')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Number of clusters: {len(set(y_pred)) - (1 if -1 in y_pred else 0)}")
print(f"Number of outliers: {np.sum(y_pred == -1)}")
```

### 7.2 Dimensionality Reduction

#### 7.2.1 PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris
iris = load_iris()
X = iris.data
y = iris.target

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Original Data')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA Transformed Data')
plt.tight_layout()
plt.show()

# Component loadings
print(f"\nComponent Loadings:")
print(pca.components_)
```

#### 7.2.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, ticks=range(10))
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization of Digits')
plt.show()
```

#### 7.2.3 UMAP (Uniform Manifold Approximation and Projection)

```python
# Install: pip install umap-learn
import umap
from sklearn.datasets import load_digits

# Load digits
digits = load_digits()
X = digits.data
y = digits.target

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, ticks=range(10))
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP Visualization of Digits')
plt.show()
```

### 7.3 Association Rule Mining

```python
# Install: pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Sample transaction data
transactions = [
    ['milk', 'bread', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'eggs'],
    ['bread', 'eggs'],
    ['milk', 'eggs']
]

# Encode transactions
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

### 7.4 Gaussian Mixture Models

```python
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate data
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Find optimal number of components using BIC
n_components_range = range(1, 8)
bics = []
aics = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X)
    bics.append(gmm.bic(X))
    aics.append(gmm.aic(X))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(n_components_range, bics, 'b-o', label='BIC')
plt.plot(n_components_range, aics, 'r-o', label='AIC')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.legend()
plt.title('Model Selection')

# Train GMM
gmm = GaussianMixture(n_components=3, random_state=42)
y_pred = gmm.fit_predict(X)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=200, marker='X')
plt.title('Gaussian Mixture Model')
plt.tight_layout()
plt.show()
```

---

## 8. Model Evaluation and Validation

### 8.1 Evaluation Metrics

#### 8.1.1 Classification Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score, log_loss)
import numpy as np

# Sample predictions
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])
y_prob = np.array([0.9, 0.3, 0.8, 0.4, 0.2, 0.9, 0.6, 0.1, 0.8, 0.7])

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision (true positives / predicted positives)
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.4f}")

# Recall (true positives / actual positives)
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.4f}")

# F1 Score (harmonic mean of precision and recall)
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Classification Report
print(f"\n{classification_report(y_true, y_pred)}")

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
print(f"ROC-AUC: {roc_auc:.4f}")

# Log Loss (cross-entropy loss)
logloss = log_loss(y_true, y_prob)
print(f"Log Loss: {logloss:.4f}")
```

#### 8.1.2 Regression Metrics

```python
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)
import numpy as np

# Sample predictions
y_true = np.array([100, 200, 300, 400, 500])
y_pred = np.array([110, 190, 310, 390, 510])

# Mean Squared Error
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.2f}")

# Root Mean Squared Error
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")

# Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")

# Mean Absolute Percentage Error
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.2%}")

# R² Score (coefficient of determination)
r2 = r2_score(y_true, y_pred)
print(f"R² Score: {r2:.4f}")
```

### 8.2 Cross-Validation

#### 8.2.1 K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Define model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Calculate scores
accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1')

print(f"Accuracy: {accuracy_scores.mean():.3f} (+/- {accuracy_scores.std():.3f})")
print(f"F1 Score: {f1_scores.mean():.3f} (+/- {f1_scores.std():.3f})")
```

#### 8.2.2 Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# For imbalanced datasets
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Stratified ensures each fold has same class distribution
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print(f"Stratified CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

#### 8.2.3 Leave-One-Out Cross-Validation

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

# For very small datasets
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
print(f"LOO CV Accuracy: {scores.mean():.3f}")
```

### 8.3 Validation Strategies

#### 8.3.1 Hold-Out Validation

```python
from sklearn.model_selection import train_test_split

# Simple hold-out
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# With validation set
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
```

#### 8.3.2 Time Series Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# For time series data
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Fold: Train size={len(train_idx)}, Test size={len(test_idx)}, Score={score:.3f}")
```

### 8.4 ROC Curves and Precision-Recall Curves

```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ROC Curve
axes[0].plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()

# Precision-Recall Curve
axes[1].plot(recall, precision, color='blue', label=f'AP = {ap:.2f}')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 8.5 Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Generate data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Calculate learning curves
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=100),
    X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
plt.plot(train_sizes, train_mean, 'b-o', label='Training score')
plt.plot(train_sizes, test_mean, 'r-o', label='Cross-validation score')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
```

### 8.6 Hyperparameter Tuning

#### 8.6.1 Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)

grid_search.fit(X, y)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.3f}")
```

#### 8.6.2 Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'learning_rate': uniform(0.01, 0.3)
}

# Random Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
)

random_search.fit(X, y)
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_:.3f}")
```

#### 8.6.3 Bayesian Optimization

```python
# Install: pip install optuna
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return score.mean()

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best trial value: {study.best_value:.3f}")
print(f"Best parameters: {study.best_params}")
```

---

## 9. Feature Engineering and Selection

### 9.1 Feature Engineering

#### 9.1.1 Numerical Feature Engineering

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create sample data
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 75000, 80000, 90000],
    'height': [170, 175, 180, 165, 185]
})

# Log transformation (for skewed distributions)
df['log_income'] = np.log1p(df['income'])

# Square root transformation
df['sqrt_income'] = np.sqrt(df['income'])

# Power transformation (Box-Cox)
from scipy import stats
df['boxcox_age'], _ = stats.boxcox(df['age'] + 1)

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['age', 'income']])
df_poly = pd.DataFrame(poly_features,
                       columns=['age', 'income', 'age^2', 'age*income', 'income^2'])

# Binning/Discretization
df['age_bin'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100],
                        labels=['young', 'adult', 'middle', 'senior'])

# Ratio features
df['income_per_age'] = df['income'] / df['age']
```

#### 9.1.2 Categorical Feature Engineering

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Create sample data
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'XL', 'S'],
    'category': ['A', 'B', 'C', 'A', 'B']
})

# Label Encoding (integer encoding)
le = LabelEncoder()
df['color_label'] = le.fit_transform(df['color'])

# One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=['color'], prefix='color')

# Ordinal Encoding (for ordered categories)
size_mapping = {'S': 1, 'M': 2, 'L': 3, 'XL': 4}
df['size_ordinal'] = df['size'].map(size_mapping)

# Target Encoding (for high cardinality)
category_means = df.groupby('category')['income'].mean() if 'income' in df.columns else {}
```

#### 9.1.3 Datetime Feature Engineering

```python
import pandas as pd

# Create datetime
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100, freq='D')
})

# Extract datetime features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter
df['dayofyear'] = df['date'].dt.dayofyear
df['weekofyear'] = df['date'].dt.isocalendar().week

# Time since reference date
reference_date = pd.Timestamp('2023-01-01')
df['days_since_ref'] = (df['date'] - reference_date).dt.days
```

#### 9.1.4 Text Feature Engineering

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

# Sample text data
df = pd.DataFrame({
    'text': [
        'The movie was great and amazing!',
        'Terrible experience, would not recommend',
        'Good film, enjoyed watching it',
        'Awful, waste of time'
    ]
})

# Bag of Words
bow = CountVectorizer()
X_bow = bow.fit_transform(df['text'])
print(f"BoW Features: {bow.get_feature_names_out()}")

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['text'])
print(f"TF-IDF Features: {tfidf.get_feature_names_out()}")

# Custom text features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))
df['has_exclamation'] = df['text'].str.contains('!').astype(int)
df['uppercase_ratio'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x))
```

### 9.2 Feature Selection

#### 9.2.1 Filter Methods

```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Univariate feature selection (ANOVA F-value)
f_scores, p_values = f_classif(X, y)
feature_importance = pd.DataFrame({
    'feature': range(10),
    'f_score': f_scores,
    'p_value': p_values
}).sort_values('f_score', ascending=False)

print("ANOVA F-scores:")
print(feature_importance)

# Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
print("\nMutual Information Scores:")
for i, score in enumerate(mi_scores):
    print(f"Feature {i}: {score:.4f}")

# Correlation with target
df = pd.DataFrame(X)
df['target'] = y
correlations = df.corr()['target'].drop('target').abs().sort_values(ascending=False)
print("\nCorrelation with target:")
print(correlations)
```

#### 9.2.2 Wrapper Methods

```python
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Recursive Feature Elimination (RFE)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(rf, n_features_to_select=5, step=1)
rfe.fit(X, y)

print(f"RFE Selected Features: {rfe.support_}")
print(f"RFE Feature Ranking: {rfe.ranking_}")

# Sequential Feature Selection
sfs = SequentialFeatureSelector(rf, n_features_to_select=5, direction='forward')
sfs.fit(X, y)
print(f"SFS Selected Features: {sfs.get_support()}")
```

#### 9.2.3 Embedded Methods

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, LassoCV
from sklearn.linear_model import Lasso

# Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importance = rf.feature_importances_
indices = np.argsort(importance)[::-1]
print("Feature Importance (Random Forest):")
for i in range(10):
    print(f"Feature {indices[i]}: {importance[indices[i]]:.4f}")

# Select from Model
selector = SelectFromModel(rf, threshold='median')
X_selected = selector.fit_transform(X, y)
print(f"\nSelected Features: {selector.get_support()}")

# L1 Regularization (Lasso)
lasso = LassoCV(cv=5)
lasso.fit(X, y)
print(f"\nLasso Selected Features: {lasso.coef_ != 0}")
```

### 9.3 Dimensionality Reduction

#### 9.3.1 PCA

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Standardize first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")

# Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(1, 21), explained_variance, alpha=0.7)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Variance per Component')

axes[1].plot(range(1, 21), cumulative_variance, 'bo-')
axes[1].axhline(y=0.95, color='r', linestyle='--')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Variance')

plt.tight_layout()
plt.show()
```

#### 9.3.2 Feature Selection with Variance Threshold

```python
from sklearn.feature_selection import VarianceThreshold

# Remove low variance features
selector = VarianceThreshold(threshold=0.1)
X_reduced = selector.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Selected: {selector.get_support()}")
```

---

# Part 1 Summary

This completes Part 1 of the Ultimate AI & ML Documentation, covering approximately 1000 lines of content. In this section, you learned:

1. **Introduction to AI/ML** - History, types of ML, and core concepts
2. **Python Programming** - Essential Python for ML development
3. **Mathematics for ML** - Linear algebra, calculus, probability, and optimization
4. **Data Science Foundations** - Data exploration, cleaning, and preprocessing
5. **Machine Learning Introduction** - Types of problems, workflow, and evaluation
6. **Supervised Learning Algorithms** - Regression, classification, and ensemble methods
7. **Unsupervised Learning** - Clustering, dimensionality reduction
8. **Model Evaluation** - Metrics, cross-validation, and hyperparameter tuning
9. **Feature Engineering** - Creating and selecting features

---

_End of Part 1 - Continue to Part 2: Intermediate Level - ML Algorithms_

---

# Part 2: Intermediate Level - Advanced ML Algorithms

---

## 10. Deep Learning Fundamentals

### 10.1 Introduction to Neural Networks

Neural networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes (neurons) organized in layers that learn to transform input data into output predictions.

#### 10.1.1 The Perceptron

The simplest neural network unit:

```python
import numpy as np

class Perceptron:
    """Single layer perceptron for binary classification."""

    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs)
        self.bias = 0

    def forward(self, x):
        """Compute the output."""
        z = np.dot(x, self.weights) + self.bias
        return 1 if z > 0 else 0

    def train(self, X, y, epochs=100, lr=0.01):
        """Train the perceptron."""
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                prediction = self.forward(xi)
                error = yi - prediction
                self.weights += lr * error * xi
                self.bias += lr * error

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate

perceptron = Perceptron(2)
perceptron.train(X, y, epochs=100)

# Test
for xi in X:
    print(f"{xi} -> {perceptron.forward(xi)}")
```

#### 10.1.2 Multi-Layer Perceptron

Adding hidden layers enables learning complex patterns:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """Multi-Layer Perceptron."""

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

# Create model
model = MLP(input_size=784, hidden_size=256, output_size=10)

# Forward pass
x = torch.randn(32, 784)  # Batch of 32, 28x28 images
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 10]
```

### 10.2 Activation Functions

Activation functions introduce non-linearity, enabling neural networks to learn complex patterns:

#### 10.2.1 Common Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Plot activation functions
x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
activations = [
    ('Sigmoid', sigmoid),
    ('Tanh', tanh),
    ('ReLU', relu),
    ('Leaky ReLU', leaky_relu),
    ('ELU', elu),
    ('Softmax (1D)', lambda x: softmax(x.reshape(1, -1))[0])
]

for ax, (name, func) in zip(axes.flatten(), activations):
    y = func(x)
    ax.plot(x, y)
    ax.set_title(name)
    ax.grid(True)
    ax.set_ylim(-1.5, 2)

plt.tight_layout()
plt.show()
```

#### 10.2.2 Choosing Activation Functions

| Layer Type              | Recommended Activation                  |
| ----------------------- | --------------------------------------- |
| Hidden Layers           | ReLU, Leaky ReLU, ELU                   |
| Output (Classification) | Softmax (multi-class), Sigmoid (binary) |
| Output (Regression)     | Linear (none)                           |

### 10.3 Forward and Backward Propagation

#### 10.3.1 Forward Propagation

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Forward pass
model = SimpleNN()
x = torch.randn(4)  # Input
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output}")
```

#### 10.3.2 Backward Propagation (Manual Implementation)

```python
import numpy as np

# Simple neural network with backpropagation
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []

        # Initialize weights with Xavier initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.activations = [X]

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = self.relu(z) if i < len(self.weights) - 1 else self.sigmoid(z)
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        deltas = [None] * len(self.weights)

        # Output layer error
        error = y - self.activations[-1]
        deltas[-1] = error * self.sigmoid_derivative(self.activations[-1])

        # Hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[i+1].dot(self.weights[i+1].T)
            deltas[i] = error * self.relu_derivative(self.activations[i+1])

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += self.activations[i].T.dot(deltas[i]) * learning_rate / m
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate / m

# Train the network
nn = NeuralNetwork([4, 8, 8, 1])
X = np.random.randn(100, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

for epoch in range(1000):
    output = nn.forward(X)
    nn.backward(X, y, learning_rate=0.1)
    if epoch % 200 == 0:
        loss = np.mean((y - output) ** 2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### 10.4 Loss Functions

#### 10.4.1 Common Loss Functions

```python
import torch
import torch.nn as nn

# Mean Squared Error (Regression)
mse_loss = nn.MSELoss()
predictions = torch.randn(10, 1)
targets = torch.randn(10, 1)
loss = mse_loss(predictions, targets)
print(f"MSE Loss: {loss.item():.4f}")

# Cross-Entropy Loss (Multi-class Classification)
ce_loss = nn.CrossEntropyLoss()
predictions = torch.randn(10, 5)  # 10 samples, 5 classes
targets = torch.randint(0, 5, (10,))
loss = ce_loss(predictions, targets)
print(f"Cross-Entropy Loss: {loss.item():.4f}")

# Binary Cross-Entropy (Binary Classification)
bce_loss = nn.BCELoss()
predictions = torch.rand(10, 1)
targets = torch.randint(0, 2, (10, 1)).float()
loss = bce_loss(predictions, targets)
print(f"BCE Loss: {loss.item():.4f}")

# Mean Absolute Error (Regression)
mae_loss = nn.L1Loss()
predictions = torch.randn(10, 1)
targets = torch.randn(10, 1)
loss = mae_loss(predictions, targets)
print(f"MAE Loss: {loss.item():.4f}")
```

#### 10.4.2 Custom Loss Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HuberLoss(nn.Module):
    """Huber loss - less sensitive to outliers than MSE."""
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, predictions, targets):
        error = predictions - targets
        abs_error = torch.abs(error)

        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic ** 2 + self.delta * linear)

class FocalLoss(nn.Module):
    """Focal loss - focuses on hard examples."""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### 10.5 Optimizers

#### 10.5.1 Gradient Descent Variants

```python
import torch
import torch.optim as optim

# Create sample model and data
model = torch.nn.Linear(10, 2)
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
criterion = nn.CrossEntropyLoss()

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (Adaptive Moment Estimation)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam with weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Adagrad
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
```

#### 10.5.2 Learning Rate Schedulers

```python
import torch.optim as optim

# Step LR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Exponential LR
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Reduce on Plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=0.1, patience=5)

# Cosine Annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# OneCycleLR (recommended for transformers)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                          steps_per_epoch=len(train_loader),
                                          epochs=10)

# In training loop
for epoch in range(num_epochs):
    train(...)
    scheduler.step()  # For Step, Exp, Cosine
    # OR
    scheduler.step(val_loss)  # For ReduceLROnPlateau
```

### 10.6 Building Your First Deep Learning Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# Generate data
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(100):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_t).float().mean()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
```

---

## 11. Neural Network Architectures

### 11.1 Convolutional Neural Networks (CNNs)

CNNs are designed for processing grid-like data, especially images:

#### 11.1.1 CNN Architecture

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# Create model
model = CNN(num_classes=10)
print(model)

# Forward pass
x = torch.randn(4, 3, 32, 32)  # Batch of 4, 32x32 RGB images
output = model(x)
print(f"Output shape: {output.shape}")  # [4, 10]
```

#### 11.1.2 CNN Components Explained

```python
# Convolutional Layer
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                  stride=1, padding=1, bias=False)

# Pooling Layer
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Batch Normalization
bn = nn.BatchNorm2d(num_features=64)

# Dropout
dropout = nn.Dropout(p=0.5)

# Padding explanation:
# - Same padding: output size = input size / stride
# - Valid padding: output size = (input - kernel) / stride + 1
# - For 32x32 input, 3x3 kernel, stride 1:
#   - Same padding: 32x32
#   - Valid padding: 30x30
```

#### 11.1.3 Popular CNN Architectures

```python
import torchvision.models as models

# ResNet-18
resnet18 = models.resnet18(pretrained=True)

# Modify for different number of classes
num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, 10)

# VGG-16
vgg16 = models.vgg16(pretrained=True)

# EfficientNet-B0
efficientnet = models.efficientnet_b0(pretrained=True)

# MobileNet-V2 (lightweight)
mobilenet = models.mobilenet_v2(pretrained=True)

# Use as feature extractor
for param in model.parameters():
    param.requires_grad = False
```

### 11.2 Recurrent Neural Networks (RNNs)

RNNs are designed for sequential data:

#### 11.2.1 Basic RNN

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """Simple RNN for sequence processing."""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, hidden = self.rnn(x)
        # out: (batch, seq_len, hidden_size)
        # hidden: (1, batch, hidden_size)
        out = self.fc(out[:, -1, :])  # Use last time step
        return out

# Example
model = SimpleRNN(input_size=10, hidden_size=32, output_size=2)
x = torch.randn(8, 20, 10)  # Batch of 8, sequence length 20, 10 features
output = model(x)
print(f"Output shape: {output.shape}")  # [8, 2]
```

#### 11.2.2 Long Short-Term Memory (LSTM)

```python
class LSTMClassifier(nn.Module):
    """LSTM for sequence classification."""

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x):
        # LSTM returns: output, (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state from both directions
        # h_n shape: (num_layers * directions, batch, hidden_size)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)

        out = self.fc(hidden)
        return out

# Example
model = LSTMClassifier(input_size=50, hidden_size=128, num_layers=2, output_size=5)
x = torch.randn(16, 30, 50)  # Batch of 16, 30 time steps, 50 features
output = model(x)
print(f"Output shape: {output.shape}")  # [16, 5]
```

#### 11.2.3 Gated Recurrent Unit (GRU)

```python
class GRUModel(nn.Module):
    """GRU - simpler than LSTM with similar performance."""

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out
```

### 11.3 Transformer Architecture

#### 11.3.1 Attention Mechanism

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Combine heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        return self.W_o(attention_output)
```

#### 11.3.2 Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    """Transformer encoder block."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class TransformerClassifier(nn.Module):
    """Full transformer classifier."""

    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, num_classes, dropout=0.1, max_len=512):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Use [CLS] token or mean pooling
        x = x.mean(dim=1)  # Mean pooling
        x = self.fc(x)

        return x
```

### 11.4 Autoencoders

#### 11.4.1 Basic Autoencoder

```python
class Autoencoder(nn.Module):
    """Autoencoder for dimensionality reduction and reconstruction."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Assuming input in [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

# Example
autoencoder = Autoencoder(input_dim=784, hidden_dim=256, latent_dim=32)
x = torch.randn(16, 784)  # 28x28 flattened images
reconstructed = autoencoder(x)
print(f"Reconstructed shape: {reconstructed.shape}")  # [16, 784]
```

#### 11.4.2 Variational Autoencoder (VAE)

```python
class VAE(nn.Module):
    """Variational Autoencoder for generation."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(reconstructed, x, mu, logvar):
    """VAE loss = Reconstruction loss + KL divergence."""
    recon_loss = F.binary_cross_entropy(reconstructed, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

---

## 12. Natural Language Processing

### 12.1 Text Preprocessing

```python
import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Complete text preprocessing pipeline."""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# Example usage
text = "This is an example sentence for preprocessing! Check out https://example.com"
processed = preprocess_text(text)
print(processed)  # ['example', 'sentence', 'preprocessing']
```

### 12.2 Text Representation

#### 12.2.1 Bag of Words

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Matrix:\n", X.toarray())

# With n-grams
vectorizer_ngram = CountVectorizer(ngram_range=(1, 2))
X_ngram = vectorizer_ngram.fit_transform(documents)
print("\nVocabulary with bigrams:", vectorizer_ngram.get_feature_names_out())
```

#### 12.2.2 TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(documents)

print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", X_tfidf.toarray())

# With custom parameters
tfidf_custom = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True  # Use 1 + log(tf)
)
```

#### 12.2.3 Word Embeddings

```python
import gensim.downloader as api

# Load pre-trained word vectors
# This downloads GloVe 100-dimensional vectors
print("Loading model...")
model = api.load('glove-wiki-gigaword-100')

# Get word vector
word_vector = model['computer']
print(f"Word vector shape: {word_vector.shape}")

# Similar words
similar = model.most_similar('computer', topn=5)
print("Similar words:", similar)

# Word analogies
# king - man + woman = queen
analogy = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print("King - man + woman =", analogy)
```

### 12.3 Text Classification

```python
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Load dataset
newsgroups = fetch_20newsgroups(subset='train')
X = newsgroups.data
y = newsgroups.target

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train_tfidf.toarray())
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val_tfidf.toarray())
y_val_t = torch.LongTensor(y_val)

# Simple classifier
class TextClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Train
model = TextClassifier(10000, 20)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        _, predicted = torch.max(val_outputs, 1)
        accuracy = (predicted == y_val_t).float().mean()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Accuracy: {accuracy:.4f}")
```

### 12.4 Sequence-to-Sequence Models

```python
class Seq2Seq(nn.Module):
    """Sequence to sequence model with attention."""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(Seq2Seq, self).__init__()

        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_size, hidden_size * 2, num_layers,
                               batch_first=True)
        self.attention = nn.Linear(hidden_size * 3, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, source, target):
        # Encode
        encoder_output, (h, c) = self.encoder(self.embedding(source))

        # Decode
        outputs = []
        decoder_input = target[:, 0].unsqueeze(1)  # <start> token

        for t in range(target.size(1)):
            decoder_output, (h, c) = self.decoder(
                self.embedding(decoder_input), (h, c)
            )

            # Attention
            energy = self.attention(torch.cat([decoder_output, encoder_output], dim=2))
            attention = torch.softmax(energy, dim=1)
            context = torch.bmm(attention.transpose(1, 2), encoder_output)

            # Combine decoder output with context
            output = self.fc(torch.cat([decoder_output, context], dim=2))
            outputs.append(output)

            # Teacher forcing
            decoder_input = target[:, t].unsqueeze(1)

        return torch.cat(outputs, dim=1)
```

---

## 13. Computer Vision

### 13.1 Image Loading and Preprocessing

```python
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load image with OpenCV
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load with PIL
image_pil = Image.open('image.jpg')
image_array = np.array(image_pil)

# Resize
resized = cv2.resize(image, (224, 224))

# Normalize (ImageNet normalization)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
normalized = (resized / 255.0 - mean) / std

# Data augmentation with torchvision
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 13.2 Object Detection

#### 13.2.1 YOLO Implementation

```python
import torch
from torchvision.models.detection import yolo_v5

# Load pretrained YOLOv5
model = yolo_v5(pretrained=True)
model.eval()

# Preprocess image
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

image = Image.open('image.jpg')
image_tensor = transform(image).unsqueeze(0)

# Detect objects
with torch.no_grad():
    predictions = model(image_tensor)

# Parse predictions
for pred in predictions:
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']

    # Filter by confidence
    mask = scores > 0.5
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
```

#### 13.2.2 Faster R-CNN

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Forward pass
with torch.no_grad():
    predictions = model(image_tensor)

# Predictions contain: boxes, labels, scores
```

### 13.3 Image Segmentation

#### 13.3.1 Semantic Segmentation with U-Net

```python
class UNet(nn.Module):
    """U-Net architecture for semantic segmentation."""

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.output(d1))
```

### 13.4 Face Recognition

```python
# Using facenet-pytorch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load pretrained models
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Detect and align face
img = Image.open('face.jpg')
img_cropped = mtcnn(img)

# Get embedding
img_embedding = resnet(img_cropped.unsqueeze(0))
print(f"Embedding shape: {img_embedding.shape}")

# Compare faces
def compare_faces(emb1, emb2, threshold=0.7):
    """Compare two face embeddings."""
    distance = (emb1 - emb2).norm().item()
    return distance < threshold, distance
```

---

## 14. Reinforcement Learning

### 14.1 Introduction to RL

Reinforcement Learning (RL) is about learning through interaction with an environment:

```python
# Key RL concepts
# - Agent: The learner/decision maker
# - Environment: What the agent interacts with
# - Action: What the agent can do
# - State: Current situation of the agent
# - Reward: Feedback from environment
# - Policy: Strategy for choosing actions
# - Value: Expected future reward
```

### 14.2 Q-Learning

```python
import numpy as np
import gym

class QLearningAgent:
    """Q-Learning agent for discrete action spaces."""

    def __init__(self, state_size, action_size, learning_rate=0.1,
                 gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """Update Q-value."""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Train on CartPole
env = gym.make('CartPole-v1')
agent = QLearningAgent(state_size=4, action_size=2)

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.learn(state, action, reward, next_state)

        state = next_state
        total_reward += reward

        if done:
            break

    agent.decay_epsilon()

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
```

### 14.3 Deep Q-Network (DQN)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    """Deep Q-Network."""

    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states),
                np.array(dones))

class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Q-network and target network
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 64

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def learn(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        # Sample from buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if random.random() < 0.01:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

### 14.4 Policy Gradient Methods

```python
class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm."""

    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class REINFORCEAgent:
    """REINFORCE agent with baseline."""

    def __init__(self, state_size, action_size):
        self.policy = PolicyNetwork(state_size, action_size)
        self.value_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

    def choose_action(self, state):
        with torch.no_grad():
            probs = self.policy(torch.FloatTensor(state))
            action = torch.multinomial(probs, 1).item()
        return action

    def update(self, rewards, states, actions):
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Get action probabilities
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        probs = self.policy(states)

        # Policy gradient loss
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        policy_loss = -(log_probs * returns).sum()

        # Value loss
        values = self.value_network(states).squeeze()
        value_loss = nn.MSELoss()(values, returns)

        # Update networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
```

---

# Part 2 Summary

This completes Part 2 of the Ultimate AI & ML Documentation, covering approximately 1000 additional lines of content. In this section, you learned:

1. **Deep Learning Fundamentals** - Neural networks, activation functions, forward/backward propagation
2. **Neural Network Architectures** - CNNs, RNNs, LSTMs, Transformers, Autoencoders
3. **Natural Language Processing** - Text preprocessing, representations, classification, seq2seq
4. **Computer Vision** - Image processing, object detection, segmentation, face recognition
5. **Reinforcement Learning** - Q-Learning, DQN, Policy Gradient methods

---

_End of Part 2 - Continue to Part 3: Advanced Level - Deep Learning & Production ML_

---

# Part 3: Advanced Deep Learning

---

## 15. Advanced Training Techniques

### 15.1 Batch Normalization

Batch normalization normalizes activations within a mini-batch to stabilize and accelerate training:

```python
import torch
import torch.nn as nn

class BatchNormMLP(nn.Module):
    """MLP with batch normalization."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BatchNormMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# For 2D inputs (batch, features)
bn1d = nn.BatchNorm1d(num_features=64)

# For 3D inputs (batch, channels, length) - e.g., CNN
bn2d = nn.BatchNorm2d(num_features=64)

# For 4D inputs (batch, channels, height, width) - e.g., 3D CNN
bn3d = nn.BatchNorm3d(num_features=64)

# How it works:
# 1. Calculate mean and variance of the batch: μ_B, σ_B²
# 2. Normalize: x̂ = (x - μ_B) / √(σ_B² + ε)
# 3. Scale and shift: y = γx̂ + β
# Learnable parameters: γ (scale), β (shift)
```

### 15.2 Dropout and Regularization

```python
import torch.nn as nn

# Standard dropout
dropout = nn.Dropout(p=0.5)  # 50% probability of dropping

# Dropout in forward pass
x = torch.randn(32, 100)
y = dropout(x)

# Dropout2d - drops entire channels (useful for CNNs)
dropout2d = nn.Dropout2d(p=0.3)

# Dropout3d - drops entire channels in 3D data
dropout3d = nn.Dropout3d(p=0.3)

# Alpha Dropout - preserves mean and variance
alpha_dropout = nn.AlphaDropout(p=0.5)

# Custom dropout with inplace operation for memory efficiency
class InplaceDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.size()) > self.p
            return x * mask.float() / (1 - self.p)
        return x
```

### 15.3 Weight Initialization

```python
import torch.nn as nn

# Xavier/Glorot initialization (for sigmoid/tanh)
def init_xavier(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# He initialization (for ReLU)
def init_he(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Orthogonal initialization (for RNNs)
def init_orthogonal(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Initialize with normal distribution
nn.init.normal_(tensor, mean=0, std=1)

# Initialize with constant
nn.init.constant_(tensor, val=0)

# Using apply method
model.apply(init_he)
```

### 15.4 Advanced Optimizers

```python
import torch.optim as optim

# Adam with AMSGrad variant
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    amsgrad=True  # Use AMSGrad variant
)

# NAdam (Adam with Nesterov momentum)
optimizer = optim.NAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    momentum_decay=0.004
)

# RAdam (Rectified Adam)
optimizer = optim.RAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Lookahead optimizer (wraps another optimizer)
base_optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Lookahead(base_optimizer, k=5, alpha=0.5)

# Gradient accumulation for large batches
accumulation_steps = 4
for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 15.5 Mixed Precision Training

```python
import torch.cuda.amp as amp

# Mixed precision training for faster training and less memory
model = model.cuda()
criterion = nn.CrossEntropyLoss()
scaler = amp.GradScaler()

for inputs, targets in dataloader:
    inputs = inputs.cuda()
    targets = targets.cuda()

    # Forward pass with autocast
    with amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()

    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()

# Custom loss scaling
scaler = amp.GradScaler(init_scale=1024)
```

### 15.6 Gradient Clipping

```python
import torch.nn.utils as utils

# Clip gradients by value
utils.clip_grad_value_(model.parameters(), clip_value=1.0)

# Clip gradients by norm (recommended)
utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)

# Global norm clipping
total_norm = utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
print(f"Total gradient norm: {total_norm:.4f}")

# Gradient penalty for WGAN
def gradient_penalty(discriminator, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data

    interpolated.requires_grad_(True)
    pred = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty
```

---

## 16. Transfer Learning

### 16.1 Using Pre-trained Models

```python
import torchvision.models as models
import torch.nn as nn

# Load pretrained models
resnet50 = models.resnet50(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
efficientnet_b0 = models.efficientnet_b0(pretrained=True)

# Feature extraction (freeze backbone)
class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

# Freeze all parameters
for param in resnet50.parameters():
    param.requires_grad = False

# Replace classifier for new task
resnet50.fc = nn.Linear(resnet50.fc.in_features, 10)

# Fine-tuning (unfreeze some layers)
for param in resnet50.layer4.parameters():
    param.requires_grad = True

# Different learning rates for different layers
optimizer = optim.Adam([
    {'params': resnet50.conv1.parameters(), 'lr': 1e-4},
    {'params': resnet50.layer4.parameters(), 'lr': 1e-3},
    {'params': resnet50.fc.parameters(), 'lr': 1e-2}
])
```

### 16.2 Transfer Learning Strategies

```python
# Strategy 1: Feature Extraction
# Use pretrained model as fixed feature extractor
feature_model = nn.Sequential(
    pretrained_model.backbone,
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
)
for param in feature_model.parameters():
    param.requires_grad = False

# Strategy 2: Fine-tuning
# Unfreeze entire model and train with low learning rate
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Strategy 3: Progressive unfreezing
# Gradually unfreeze layers during training
def unfreeze_schedule(epoch):
    if epoch == 10:
        unfreeze_layer(model.layer3)
    elif epoch == 20:
        unfreeze_layer(model.layer4)
    elif epoch == 30:
        unfreeze_layer(model.layer2)

# Domain adaptation with adversarial training
class DomainAdaptation(nn.Module):
    def __init__(self, feature_extractor, label_classifier, domain_classifier):
        super().__init__()
        self.features = feature_extractor
        self.label_classifier = label_classifier
        self.domain_classifier = domain_classifier

    def forward(self, x, alpha=1.0):
        features = self.features(x)
        class_output = self.label_classifier(features)

        # Gradient reversal for domain adaptation
        domain_output = self.domain_classifier(features.detach())
        domain_output_adapted = self.gradient_reversal(domain_output, alpha)

        return class_output, domain_output_adapted

    def gradient_reversal(self, x, alpha):
        return (x * alpha).requires_grad_(True)
```

### 16.3 Model Ensemble

```python
# Ensemble multiple models for better performance
class Ensemble(nn.Module):
    def __init__(self, models, method='voting'):
        super().__init__()
        self.models = models
        self.method = method

    def forward(self, x):
        outputs = [model(x) for model in self.models]

        if self.method == 'voting':
            # Hard voting
            votes = torch.stack(outputs, dim=0)
            return votes.mode(dim=0)[0]

        elif self.method == 'averaging':
            # Average probabilities
            probs = [torch.softmax(out, dim=1) for out in outputs]
            avg_probs = torch.stack(probs, dim=0).mean(dim=0)
            return torch.log(avg_probs)

        elif self.method == 'weighted':
            weights = [0.3, 0.5, 0.2]  # Learnable or predefined
            probs = [torch.softmax(out, dim=1) for out in outputs]
            weighted_probs = sum(w * p for w, p in zip(weights, probs))
            return torch.log(weighted_probs)

# Use ensemble
ensemble = Ensemble([model1, model2, model3], method='averaging')
```

---

## 17. Generative Models

### 17.1 Generative Adversarial Networks (GANs)

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    """Generator network for GAN."""

    def __init__(self, latent_dim, img_channels, img_size):
        super().__init__()
        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)
        return img

class Discriminator(nn.Module):
    """Discriminator network for GAN."""

    def __init__(self, img_channels, img_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(img_channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Training loop
def train_gan(generator, discriminator, dataloader, latent_dim, epochs):
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for imgs in dataloader:
            batch_size = imgs.size(0)

            # Ground truths
            valid = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.zero_grad()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
```

### 17.2 StyleGAN

```python
class StyleGANGenerator(nn.Module):
    """StyleGAN-inspired generator with mapping network."""

    def __init__(self, latent_dim=512, hidden_dim=512, img_channels=3):
        super().__init__()

        # Mapping network
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Synthesis network (progressively growing)
        self.synthesis = SynthesisNetwork(latent_dim, img_channels)

    def forward(self, z, noise=None):
        w = self.mapping(z)
        img = self.synthesis(w, noise)
        return img

class SynthesisBlock(nn.Module):
    """Single synthesis block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.noise1 = NoiseInjection()
        self.noise2 = NoiseInjection()
        self.activation = nn.LeakyReLU(0.2)
        self.style1 = StyleMod(in_channels)
        self.style2 = StyleMod(out_channels)

    def forward(self, x, w, noise):
        x = self.conv1(x)
        x = self.noise1(x, noise)
        x = self.style1(x, w)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.noise2(x, noise)
        x = self.style2(x, w)
        x = self.activation(x)
        return x
```

### 17.3 Diffusion Models

```python
class DiffusionModel(nn.Module):
    """DDPM (Denoising Diffusion Probabilistic Models)."""

    def __init__(self, image_channels, image_size, hidden_dims=[64, 128, 256, 512]):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )

        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = image_channels
        for h in hidden_dims:
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_ch, h, 4, 2, 1),
                nn.BatchNorm2d(h),
                nn.SiLU()
            ))
            in_ch = h

        # Decoder (with skip connections)
        self.decoder = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 4, 2, 1),
                nn.BatchNorm2d(hidden_dims[i-1]),
                nn.SiLU()
            ))

        self.final = nn.Conv2d(hidden_dims[0], image_channels, 3, padding=1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(1))

        # Encoder with time conditioning
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            h = h + t_emb.unsqueeze(2).unsqueeze(3)

        # Decoder
        for block in self.decoder:
            h = block(h)

        return self.final(h)

# Training diffusion model
def train_diffusion(model, dataloader, optimizer, device, timesteps=1000):
    model.train()
    for images in dataloader:
        images = images.to(device)
        batch_size = images.size(0)

        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(images)

        # Add noise to images
        alpha_bar = get_alpha_bar(t)  # Precomputed schedule
        noisy_images = torch.sqrt(alpha_bar) * images + torch.sqrt(1 - alpha_bar) * noise

        # Predict noise
        predicted_noise = model(noisy_images, t.float() / timesteps)

        # Compute loss
        loss = nn.functional.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 18. Model Optimization and Compression

### 18.1 Quantization

```python
import torch.quantization as tq

# Post-training quantization
model = Model()
model.eval()

# Specify quantization configuration
model.qconfig = tq.get_default_qconfig('fbgemm')
tq.prepare(model, inplace=True)

# Calibrate with representative dataset
with torch.no_grad():
    for data in calibration_loader:
        model(data)

# Convert to quantized model
quantized_model = tq.convert(model, inplace=False)

# Dynamic quantization (simpler, for inference)
# Linear layers become int8
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear: torch.quantization.default_dynamic_qconfig},
    dtype=torch.qint8
)

# Quantization-aware training
class QuantAwareTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn(self.conv(x)))
        x = self.dequant(x)
        return x
```

### 18.2 Pruning

```python
import torch.nn.utils.prune as prune

# Magnitude pruning - remove weights with smallest absolute value
prune.l1_unstructured(model.conv1, name='weight', amount=0.5)

# Random pruning
prune.random_unstructured(model.conv1, name='weight', amount=0.3)

# Structured pruning (remove entire channels/neurons)
prune.ln_structured(
    model.conv1, name='weight', amount=0.5, n=2, dim=0
)

# Global pruning across entire model
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2
)

# Remove pruning reparameterization (make it permanent)
for module, name in parameters_to_prune:
    prune.remove(module, name)

# Check sparsity
sparsity = 1.0 - (model.conv1.weight == 0).float().mean()
print(f"Sparsity: {sparsity:.2%}")
```

### 18.3 Knowledge Distillation

```python
class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation."""

    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft label loss (knowledge distillation)
        soft_loss = self.kl_loss(
            torch.log_softmax(student_logits / self.temperature, dim=1),
            torch.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Training with distillation
teacher = load_pretrained_teacher()
student = create_student_model()

teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

criterion = DistillationLoss(temperature=4.0, alpha=0.7)

for images, labels in dataloader:
    # Get teacher predictions
    with torch.no_grad():
        teacher_logits = teacher(images)

    # Get student predictions
    student_logits = student(images)

    # Compute loss
    loss = criterion(student_logits, teacher_logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 18.4 Neural Architecture Search (NAS)

```python
# Simple NAS with random search
import random

class Architecture:
    def __init__(self):
        self.operations = ['conv_3x3', 'conv_5x5', 'depthwise_conv', 'max_pool', 'avg_pool', 'skip']
        self.edges = {}  # (i, j): operation

    def mutate(self):
        # Randomly change one operation
        new_arch = Architecture()
        new_arch.edges = self.edges.copy()

        # Add, remove, or modify edge
        action = random.choice(['add', 'remove', 'modify'])
        if action == 'add':
            i, j = random.randint(0, 5), random.randint(0, 5)
            new_arch.edges[(i, j)] = random.choice(self.operations)
        elif action == 'remove':
            if self.edges:
                key = random.choice(list(self.edges.keys()))
                del new_arch.edges[key]
        else:
            if self.edges:
                key = random.choice(list(self.edges.keys()))
                new_arch.edges[key] = random.choice(self.operations)

        return new_arch

    def train_and_evaluate(self):
        model = build_model(self)
        return train_and_evaluate(model)

# Evolutionary NAS
population = [Architecture() for _ in range(20)]
for generation in range(50):
    # Evaluate all architectures
    fitness = [arch.train_and_evaluate() for arch in population]

    # Select best
    sorted_pop = [x for _, x in sorted(zip(fitness, population), reverse=True)]
    survivors = sorted_pop[:10]

    # Create new population
    new_population = survivors.copy()
    while len(new_population) < 20:
        parent = random.choice(survivors)
        child = parent.mutate()
        new_population.append(child)

    population = new_population
```

---

## 19. Deployment and Production

### 19.1 Model Export

```python
import torch
import onnx
import pickle

# Save PyTorch model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'model_checkpoint.pt')

# Save only model weights
torch.save(model.state_dict(), 'model_weights.pt')

# Load model
model = ModelClass()
model.load_state_dict(torch.load('model_weights.pt'))

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Verify ONNX model
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)

# Save with pickle (not recommended for production)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 19.2 ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession(
    'model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
def predict(session, input_data):
    input_data = input_data.astype(np.float32)
    output = session.run([output_name], {input_name: input_data})
    return output[0]

# Batch inference
def predict_batch(session, inputs):
    inputs = np.array(inputs).astype(np.float32)
    outputs = session.run([output_name], {input_name: inputs})
    return outputs[0]

# Warmup
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
for _ in range(10):
    predict(session, dummy_input)
```

### 19.3 Flask API for ML Model

```python
from flask import Flask, request, jsonify
import torch
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# Load model at startup
session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Preprocess
    input_data = np.array(data['input']).astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    # Inference
    output = session.run([output_name], {input_name: input_data})[0]

    # Postprocess
    prediction = {
        'class': int(np.argmax(output)),
        'probability': float(np.max(output))
    }

    return jsonify(prediction)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    inputs = np.array(data['inputs']).astype(np.float32)

    outputs = session.run([output_name], {input_name: inputs})[0]

    predictions = [
        {'class': int(np.argmax(out)), 'probability': float(np.max(out))}
        for out in outputs
    ]

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 19.4 Docker for ML Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app
COPY model.onnx .
COPY app.py .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - '5000:5000'
    environment:
      - MODEL_PATH=/app/model.onnx
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
    restart: unless-stopped
```

### 19.5 TensorFlow Serving

```python
# Save model in SavedModel format
import tensorflow as tf

model = create_model()
model.save('saved_model/my_model/1')

# Or export for TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open('model.tflite', 'wb').write(tflite_model)

# Run TensorFlow Serving with Docker
# docker run -t -p 8501:8501 \
#   --mount type=bind,source=/path/to/models,target=/models \
#   -e MODEL_NAME=my_model \
#   tensorflow/serving
```

### 19.6 TorchServe

```yaml
# config.properties
model_store=/model_store
models=my_model.mar

# Register model
torchserve --start --model-store /model_store --models my_model.mar

# Make predictions
import requests

payload = {
    "instances": [input_data.tolist()]
}
response = requests.post(
    "http://localhost:8080/predictions/my_model",
    json=payload
)
print(response.json())
```

---

## 20. MLOps Fundamentals

### 20.1 Experiment Tracking

```python
# Using MLflow
import mlflow

mlflow.set_experiment("image_classification")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 100)

    # Log metrics
    for epoch in range(100):
        train_loss = train_one_epoch()
        val_loss = validate()
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")

# Using Weights & Biases
import wandb

wandb.init(project="image_classification", entity="your_username")

config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "model": "resnet50"
}
wandb.config = config

for epoch in range(100):
    train_loss = train_one_epoch()
    val_loss = validate()

    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch
    })

wandb.finish()
```

### 20.2 Data Versioning

```python
# Using DVC (Data Version Control)
# dvc.yaml
stages:
  preprocess:
    cmd: python preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/
    params:
      - preprocess
    outs:
      - data/processed/

  train:
    cmd: python train.py
    deps:
      - src/train.py
      - data/processed/
    params:
      - train
    outs:
      - models/model.pt
    metrics:
      - metrics.json:
          cache: false

# Using Delta Lake for data versioning
import deltalake

# Create delta table
dt = deltalake.DeltaTable.create(
    table_uri="data/table",
    schema=schema,
    partition_by=["date"]
)

# Write with version
dt.write(data, mode="overwrite", version=1)

# Read specific version
dt = deltalake.DeltaTable.from_path("data/table", version=2)
df = dt.to_pandas()
```

### 20.3 CI/CD for ML

```yaml
# .github/workflows/ml-ci-cd.yml
name: ML CI/CD

on:
  push:
    branches: [main]
    paths:
      - '**.py'
      - 'data/**'
      - 'models/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

      - name: Lint
        run: |
          flake8 src/
          mypy src/

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Train model
        run: python train.py

      - name: Evaluate
        run: python evaluate.py

      - name: Upload model
        uses: actions/upload-artifact@v2
        with:
          name: model
          path: models/model.pt

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Download model
        uses: actions/download-artifact@v2
        with:
          name: model

      - name: Deploy to production
        run: |
          docker build -t myapp:${{ github.sha }} .
          docker push myapp:${{ github.sha }}
```

### 20.4 Model Monitoring

```python
# Prometheus metrics for model monitoring
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
prediction_count = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
feature_drift = Gauge('feature_drift', 'Feature drift score')

# Track predictions
@app.route('/predict')
def predict():
    start_time = time.time()

    # Make prediction
    result = model.predict(input_data)

    # Record metrics
    prediction_count.inc()
    prediction_latency.observe(time.time() - start_time)

    return result

# Scheduled drift detection
def check_data_drift():
    reference_data = load_reference_data()
    current_data = load_current_data()

    # Calculate drift
    drift_score = calculate_drift(reference_data, current_data)
    feature_drift.set(drift_score)

    if drift_score > threshold:
        alert_team(f"Drift detected: {drift_score:.3f}")

# Schedule drift check
schedule.every(1).hours.do(check_data_drift)
```

---

# Part 3 Summary

This completes Part 3 of the Ultimate AI & ML Documentation, covering approximately 1000 additional lines of advanced content. In this section, you learned:

1. **Advanced Training Techniques** - Batch normalization, dropout, weight initialization, advanced optimizers
2. **Transfer Learning** - Using pretrained models, fine-tuning strategies, model ensembling
3. **Generative Models** - GANs, StyleGAN, Diffusion models
4. **Model Optimization** - Quantization, pruning, knowledge distillation, NAS
5. **Deployment and Production** - Model export, ONNX, Flask API, Docker, TensorFlow Serving
6. **MLOps Fundamentals** - Experiment tracking, data versioning, CI/CD, monitoring

---

_End of Part 3 - Continue to Part 4: Modern AI & Generative AI_

---

# Part 4: Modern AI - LLMs, RAG, Fine Tuning & Agentic AI

---

## 21. Large Language Models (LLMs)

### 21.1 Introduction to LLMs

Large Language Models are deep learning models trained on vast amounts of text data, capable of understanding and generating human-like text. Models like GPT, Claude, and Gemini have revolutionized NLP.

#### 21.1.1 Transformer Architecture for LLMs

```python
# LLM Architecture Components
class LLMTransformer(nn.Module):
    """
    Large Language Model using transformer decoder architecture.
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (share embeddings with LM head)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # Create position IDs
        position_ids = torch.arange(seq_len).expand(batch_size, -1).to(input_ids.device)

        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = x * math.sqrt(self.d_model)

        # Causal mask (prevent attending to future tokens)
        causal_mask = self.create_causal_mask(seq_len, input_ids.device)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits

    def create_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
```

#### 21.1.2 Using Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pretrained LLM
model_name = "gpt2"  # or "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto"  # Automatically distribute across GPUs
)

# Generate text
def generate_text(prompt, max_length=100, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
prompt = "Once upon a time in a distant galaxy"
generated = generate_text(prompt, max_length=150)
print(generated)
```

### 21.2 Tokenization

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Basic tokenization
text = "Hello, how are you today?"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)

print(f"Original: {text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
print(f"Token count: {len(tokens)}")

# Word-level vs subword tokenization
text = "Artificial Intelligence is transforming the world"

# Word tokenization
word_tokens = text.split()
print(f"Word tokens: {word_tokens}")

# Subword tokenization (BPE, WordPiece, SentencePiece)
subword_tokens = tokenizer.tokenize(text)
print(f"Subword tokens: {subword_tokens}")

# Batch tokenization
sentences = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "Transformers revolutionized NLP."
]

encoded = tokenizer(sentences, padding=True, truncation=True, max_length=50, return_tensors="pt")
print(f"Encoded: {encoded}")
```

### 21.3 LLM Inference Optimization

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model in optimized mode
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Enable optimizations
model.eval()  # Set to evaluation mode

# Quantization (reduce model size)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True  # Use 8-bit quantization
)

model_quantized = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# KV Cache optimization for faster generation
def generate_with_cache(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Use past key values for faster generation
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        use_cache=True,  # Enable KV cache
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Batch inference for multiple prompts
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is deep learning?"
]

def batch_generate(prompts, max_length=50):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,  # Greedy for speed
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

results = batch_generate(prompts)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

---

## 22. Fine Tuning LLMs

### 22.1 Full Fine Tuning

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
dataset = load_dataset("imdb", split="train")
dataset = dataset.train_test_split(test_size=0.1)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  # Mixed precision
    gradient_accumulation_steps=4,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Train
trainer.train()

# Save model
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
```

### 22.2 Parameter-Efficient Fine Tuning (PEFT)

#### 22.2.1 LoRA (Low-Rank Adaptation)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Rank of adaptation matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],  # Which layers to adapt
    bias="none",
    inference_mode=False
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Output: trainable params: 1,536,768 || all params: 124,590,080 || trainable%: 1.23

# Training with LoRA
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()

# Save only LoRA weights
model.save_pretrained("./lora-model")
```

#### 22.2.2 QLoRA (Quantized LoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Quantization config for 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Train - uses much less GPU memory!
```

#### 22.2.3 IA3 (Adapter-based Fine Tuning)

```python
from peft import IA3Config, get_peft_model

# IA3 config - adds learned vectors to key layers
ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out"],
    feedforward_modules=["fc_in", "fc_out"],
)

model = get_peft_model(model, ia3_config)
model.print_trainable_parameters()
```

### 22.3 RLHF (Reinforcement Learning from Human Feedback)

```python
# RLHF Pipeline using TRL library
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DPOTrainer, ORPOTrainer

# Step 1: Supervised Fine Tuning (SFT)
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=sft_dataset,
    tokenizer=tokenizer,
    max_seq_length=512,
)

sft_trainer.train()

# Step 2: Reward Model Training
reward_model = AutoModelForCausalLM.from_pretrained("gpt2")
# Train on human preferences (chosen > rejected)

# Step 3: PPO Training (Reinforcement Learning)
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.4e-5,
    batch_size=4,
    forward_batch_size=1,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    reward_model=reward_model,
    tokenizer=tokenizer,
)

# Fine-tune with PPO
ppo_trainer.train()

# Save final model
model.save_pretrained("./rlhf-model")
```

---

## 23. Retrieval-Augmented Generation (RAG)

### 23.1 RAG Architecture

```python
# RAG System Components

class RAGSystem:
    """
    Retrieval-Augmented Generation system.
    Combines retrieval (search) with LLM generation.
    """

    def __init__(self, model_name="gpt2", retriever_type="faiss"):
        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Initialize retriever
        if retriever_type == "faiss":
            self.retriever = FAISSRetriever()
        elif retriever_type == "bm25":
            self.retriever = BM25Retriever()
        elif retriever_type == "huggingface":
            self.retriever = HuggingFaceRetriever()

        self.document_store = None

    def index_documents(self, documents):
        """Index documents for retrieval."""
        # Create embeddings
        embeddings = self.retriever.encode(documents)

        # Build index
        self.document_store = self.retriever.build_index(embeddings)

        self.documents = documents

    def retrieve(self, query, top_k=3):
        """Retrieve relevant documents."""
        query_embedding = self.retriever.encode([query])
        scores, indices = self.document_store.search(query_embedding, k=top_k)

        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs, scores[0]

    def generate(self, query, retrieved_docs):
        """Generate answer with retrieved context."""
        # Build prompt with context
        context = "\n\n".join(retrieved_docs)
        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {query}

Answer:"""

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def query(self, question, top_k=3):
        """Full RAG pipeline."""
        # 1. Retrieve relevant documents
        retrieved_docs, scores = self.retrieve(question, top_k)

        # 2. Generate answer with context
        answer = self.generate(question, retrieved_docs)

        return answer, retrieved_docs, scores
```

### 23.2 Vector Databases

```python
# Using FAISS for vector similarity search
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def encode(self, texts):
        """Create embeddings for texts."""
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return embeddings.astype('float32')

    def build_index(self, embeddings):
        """Build FAISS index."""
        dimension = embeddings.shape[1]

        # Use HNSW for efficient search
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.add(embeddings)

        return self.index

    def search(self, query_embedding, k=5):
        """Search for similar documents."""
        scores, indices = self.index.search(query_embedding, k)
        return scores, indices

# Using ChromaDB (popular vector database)
import chromadb
from chromadb.config import Settings

class ChromaRetriever:
    def __init__(self, collection_name="documents"):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(self, documents, ids=None):
        """Add documents to collection."""
        if ids is None:
            ids = [str(i) for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            ids=ids
        )

    def search(self, query, n_results=3):
        """Search for similar documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return results['documents'][0], results['distances'][0]

# Using Pinecone (cloud vector database)
from pinecone import Pinecone, ServerlessSpec

class PineconeRetriever:
    def __init__(self, api_key, environment, index_name="rag-index"):
        self.pc = Pinecone(api_key=api_key)

        # Create index if not exists
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )

        self.index = self.pc.Index(index_name)

    def upsert_vectors(self, vectors, ids, documents):
        """Add vectors to index."""
        self.index.upsert(
            vectors=list(zip(ids, vectors, [{"text": doc} for doc in documents]))
        )

    def search(self, query_vector, top_k=5):
        """Search for similar vectors."""
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        return results['matches']
```

### 23.3 Building Complete RAG Pipeline

```python
# Complete RAG pipeline with LangChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class LangChainRAG:
    def __init__(self, openai_api_key=None, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # LLM (use OpenAI or local)
        if openai_api_key:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=openai_api_key,
                temperature=0.7
            )

        self.vectorstore = None

    def load_documents(self, file_paths):
        """Load documents from files."""
        documents = []

        for path in file_paths:
            if path.endswith('.pdf'):
                loader = PDFLoader(path)
            else:
                loader = TextLoader(path)

            docs = loader.load()
            documents.extend(docs)

        return documents

    def create_index(self, documents):
        """Create vector index from documents."""
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Create vector store
        self.vectorstore = FAISS.from_documents(
            chunks,
            self.embeddings
        )

        return len(chunks)

    def retrieve(self, query, k=4):
        """Retrieve relevant documents."""
        if not self.vectorstore:
            raise ValueError("No index created. Call create_index first.")

        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

    def generate(self, query, retrieved_docs):
        """Generate answer using RAG."""
        # Build context from retrieved docs
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Create prompt
        system_message = """You are a helpful AI assistant. Use the following context to answer the user's question.
        If you cannot find the answer in the context, say so."""

        human_message = f"Context:\n{context}\n\nQuestion: {query}"

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]

        # Generate
        response = self.llm(messages)
        return response.content

    def query(self, question, k=4):
        """Full RAG query pipeline."""
        # Retrieve
        docs = self.retrieve(question, k)

        # Generate
        answer = self.generate(question, docs)

        return answer, docs

# Example usage
rag = LangChainRAG(openai_api_key="your-api-key")

# Load documents
docs = rag.load_documents(["document1.txt", "document2.pdf"])
print(f"Loaded {len(docs)} documents")

# Create index
num_chunks = rag.create_index(docs)
print(f"Created index with {num_chunks} chunks")

# Query
answer, sources = rag.query("What is machine learning?")
print(f"Answer: {answer}")
print(f"Sources: {len(sources)} documents")
```

---

## 24. Prompt Engineering

### 24.1 Basic Prompting Techniques

```python
# Zero-shot prompting - no examples
def zero_shot_prompt(model, prompt):
    """Use model without any examples."""
    response = model.generate(prompt)
    return response

# Example
prompt = "Classify this sentiment: The movie was absolutely amazing!"
# Output: Positive

# One-shot prompting - one example
def one_shot_prompt(model, prompt, example):
    """Use model with one example."""
    full_prompt = f"{example}\n\n{prompt}"
    return model.generate(full_prompt)

# Example
example = "The food was terrible. -> Negative"
prompt = "The service was excellent. ->"
# Output: Positive

# Few-shot prompting - multiple examples
def few_shot_prompt(model, prompt, examples):
    """Use model with multiple examples."""
    few_shot_prompt = "\n\n".join(examples + [prompt])
    return model.generate(few_shot_prompt)

examples = [
    "The cat sat on the mat. -> Noun phrase: cat, mat",
    "The dog ran in the park. -> Noun phrase: dog, park",
]
prompt = "The bird flew across the sky. ->"

# Output: Noun phrase: bird, sky
```

### 24.2 Advanced Prompting

```python
# Chain of Thought (CoT) prompting
def cot_prompt(model, problem):
    """Prompt with chain of thought reasoning."""
    prompt = f"""Solve this problem step by step:

Problem: {problem}

Let's think step by step:"""
    return model.generate(prompt)

# Example
problem = "If there are 5 birds and you shoot 2, how many are left?"
response = cot_prompt(model, problem)
# Output: "There are 5 birds. You shoot 2 birds. The 2 birds die and are gone.
# The remaining birds would fly away when they hear the gunshot. So 0 birds are left."

# Self-Consistency (multiple reasoning paths)
def self_consistency_prompt(model, problem, num_samples=5):
    """Generate multiple solutions and take majority vote."""
    responses = []

    for _ in range(num_samples):
        prompt = f"""Solve this problem. Show your reasoning:

{problem}"""

        response = model.generate(prompt, temperature=0.8)
        responses.append(response)

    # In practice, you'd parse and vote
    return responses

# Tree of Thoughts prompting
def tree_of_thought_prompt(model, problem, depth=3):
    """Explore multiple reasoning paths."""
    def expand_thought(thought, depth):
        if depth == 0:
            return [thought]

        prompt = f"""Continue this reasoning path:
{thought}

What are the next possible steps?"""

        next_steps = model.generate(prompt, n=3)  # Generate 3 branches

        expanded = []
        for step in next_steps:
            expanded.extend(expand_thought(f"{thought}\n{step}", depth-1))

        return expanded

    initial_prompt = f"""Start solving this problem:
{problem}

Think of multiple approaches."""

    thoughts = model.generate(initial_prompt, n=3)
    all_thoughts = []

    for thought in thoughts:
        all_thoughts.extend(expand_thought(thought, depth))

    return all_thoughts
```

### 24.3 Prompt Templates

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Simple template
template = """You are a {role} assistant.

User's question: {question}

Provide a helpful and {tone} response."""

prompt = PromptTemplate(
    template=template,
    input_variables=["role", "question", "tone"],
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Use chain
response = chain.run({
    "role": "technical",
    "question": "What is Python?",
    "tone": "professional"
})

# Few-shot with examples
few_shot_template = """Classify the sentiment of each text.

Example 1: "I love this product!" -> Positive
Example 2: "This is terrible." -> Negative
Example 3: "It works as expected." -> Neutral

Now classify: {text}"""

prompt = PromptTemplate(
    template=few_shot_template,
    input_variables=["text"]
)

# Chat prompt template
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant specialized in {topic}."),
    ("human", "{user_input}"),
    ("ai", "{ai_response}"),  # Few-shot example
    ("human", "{new_question}"),
])

messages = chat_template.format_messages(
    role="helpful",
    topic="technology",
    user_input="What is AI?",
    ai_response="Artificial Intelligence (AI) is...",
    new_question="What is ML?"
)
```

### 24.4 Structured Output Prompting

```python
# Prompt for JSON output
json_template = """Generate a JSON response with the following structure:

{{
    "name": " person's name",
    "age": number,
    "skills": ["skill1", "skill2"],
    "experience": [
        {{
            "company": "company name",
            "years": number
        }}
    ]
}}

Based on this resume text:
{resume_text}

Generate the JSON:"""

# Pydantic model for structured output
from pydantic import BaseModel
from typing import List, Optional

class Experience(BaseModel):
    company: str
    years: int
    role: str

class Person(BaseModel):
    name: str
    age: int
    skills: List[str]
    experience: List[Experience]
    email: Optional[str] = None

# Use with langchain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

parser = PydanticOutputParser(pydantic_object=Person)

prompt = PromptTemplate(
    template="Extract person information from: {text}\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Generate
text = "John Doe is a 30-year-old software engineer with 8 years experience at Google and Microsoft. He knows Python, Java, and JavaScript. His email is john@example.com."

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(text)
person = parser.parse(result)

print(person.name)  # John Doe
print(person.skills)  # ['Python', 'Java', 'JavaScript']
```

---

## 25. LangChain & AI Agents

### 25.1 LangChain Basics

```python
# LangChain Components
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.agents import AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Initialize LLM
llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# Simple chain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms for a 5-year-old."
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="quantum physics")
print(result)

# Sequential chains
chain1 = LLMChain(llm=llm, prompt=first_prompt)
chain2 = LLMChain(llm=llm, prompt=second_prompt)

combined = SimpleSequentialChain(chains=[chain1, chain2])
final_result = combined.run(topic)
```

### 25.2 Building AI Agents

```python
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.tools import Tool

# Define tools for the agent
def search_wikipedia(query):
    """Search Wikipedia for information."""
    # Implementation
    return "Result from Wikipedia..."

def calculate(expression):
    """Evaluate mathematical expressions."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"

# Create tools
tools = [
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for looking up factual information."
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Useful for mathematical calculations."
    )
]

# Create prompt with tools
prefix = """You have access to the following tools:

{tools}

Use the following format:

Thought: consider what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this can repeat)

Begin!"""

suffix = """Question: {input}

{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)

# Create agent
agent = ZeroShotAgent(llm=llm, prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# Run agent
result = agent_executor.run("What is the capital of France? Also calculate 15 * 23.")
print(result)
```

### 25.3 Memory in Agents

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# Buffer memory - stores full conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Summary memory - summarizes old messages
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# Entity memory - remembers entities
from langchain.memory import EntityMemory

memory = EntityMemory(
    llm=llm,
    memory_key="entities",
    entity_store="entity_store"
)

# Use with agent
from langchain.agents import AgentExecutor, ConversationAgent

agent = ConversationAgent(
    llm=llm,
    prompt=prompt,
    tools=tools,
    memory=memory
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Multi-line conversation
agent_executor.run("My name is John.")
agent_executor.run("What's my name?")
```

### 25.4 Custom Tools and Functions

```python
from langchain.tools import Tool
from langchain.utilities import PythonREPL

# Python code execution tool
python_repl = PythonREPL()

python_tool = Tool(
    name="Python_REPL",
    func=python_repl.run,
    description="Use this to execute Python code. Input should be valid Python code."
)

# Custom API tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    # API call would go here
    return f"Weather in {location}: 22°C, sunny"

weather_tool = Tool(
    name="GetWeather",
    func=get_weather,
    description="Get current weather for a city or location."
)

# Database query tool
def query_database(query: str) -> str:
    """Query the database. Returns results as JSON."""
    # Implementation
    return '[{"id": 1, "name": "Product A"}, {"id": 2, "name": "Product B"}]'

db_tool = Tool(
    name="Database",
    func=query_database,
    description="Query the product database. Input is a SQL-like query."
)

# Web search tool
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="WebSearch",
    func=search.run,
    description="Search the web for current information."
)

# Combined tools
tools = [python_tool, weather_tool, db_tool, search_tool]
```

---

## 26. Emerging Topics

### 26.1 Multimodal Models

```python
# Using Claude or GPT-4V for multimodal input
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

# Load multimodal model
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = AutoModelForVision2Seq.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Process image and text
def describe_image(image_path, question):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=question,
        images=image,
        return_tensors="pt"
    ).to(model.device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

# Example
description = describe_image("photo.jpg", "What's happening in this image?")
print(description)

# Generate image from text using Stable Diffusion
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")

image = pipe(
    prompt="A serene mountain landscape at sunset, digital art",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save("generated_landscape.png")
```

### 26.2 AI Safety and Alignment

```python
# Implementing safety filters
class SafetyFilter:
    def __init__(self, llm):
        self.llm = llm
        self.blocked_topics = ["violence", "harmful content", "illegal activities"]

    def check_prompt(self, prompt: str) -> bool:
        """Check if prompt is safe."""
        for topic in self.blocked_topics:
            if topic.lower() in prompt.lower():
                return False
        return True

    def check_output(self, output: str) -> str:
        """Check and sanitize output."""
        # Implement content filtering
        return output

# Implementing guardrails
class Guardrails:
    def __init__(self):
        self.input_validator = InputValidator()
        self.output_validator = OutputValidator()

    def validate_input(self, prompt: str) -> dict:
        """Validate user input."""
        return {
            "valid": len(prompt) < 10000,
            "language": "en",
            "contains_pii": False
        }

    def validate_output(self, output: str) -> dict:
        """Validate model output."""
        return {
            "valid": True,
            "toxicity_score": 0.1,
            "contains_sensitive_info": False
        }

# Implementing output moderation
from transformers import pipeline

moderation = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")

def moderate(text: str) -> dict:
    """Check content for policy violations."""
    result = moderation(text)

    return {
        "flagged": result[0]["label"] == "hate",
        "scores": result[0]["score"]
    }
```

---

# Part 4 Summary

This completes Part 4 of the AI & ML Documentation, covering modern AI topics. In this section, you learned:

1. **Large Language Models (LLMs)** - Architecture, tokenization, inference optimization
2. **Fine Tuning** - Full fine tuning, LoRA, QLoRA, IA3, RLHF
3. **Retrieval-Augmented Generation (RAG)** - Vector databases, embeddings, complete RAG pipelines
4. **Prompt Engineering** - Zero-shot, Chain of Thought, Tree of Thoughts, structured outputs
5. **LangChain & Agents** - Building AI agents with tools and memory
6. **Emerging Topics** - Multimodal models, AI safety, guardrails

---

_End of Part 4 - The Ultimate AI & ML Documentation is Complete!_
