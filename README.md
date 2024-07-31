# Binary Classification Algorithm

This repository contains a simple implementation of a binary classification algorithm using logistic regression. The algorithm is implemented in Python without relying on machine learning libraries.

## Table of Contents

1. [Introduction](#introduction)
2. [Algorithm Overview](#algorithm-overview)
3. [Implementation](#implementation)
4. [Usage](#usage)
5. [Contributing](#contributing)

## Introduction

Binary classification is a fundamental task in machine learning where the goal is to categorize input data into one of two classes. This implementation focuses on logistic regression, a simple yet effective algorithm for binary classification.

## Algorithm Overview

Logistic regression works by modeling the probability that an input belongs to the positive class. The core of the algorithm is as follows:

1. **Model**: The logistic function (sigmoid) is used to map input features to a probability:
   
   P(y=1|x) = 1 / (1 + e^(-z))
   
   where z = w^T x + b

   Here, w are the weights, x is the input feature vector, and b is the bias term.

2. **Training**: The weights are learned by minimizing the log loss (cross-entropy):
   
   L(w) = -1/N * Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)]

   This is typically done using gradient descent:

   w = w - α * ∂L/∂w

   where α is the learning rate.

3. **Prediction**: For a new input x, we compute P(y=1|x) and classify as positive if this probability is ≥ 0.5, and negative otherwise.

## Implementation

The algorithm is implemented in a single Python file `binary_classification.ipynb`:

## Usage 

Ensure you have installed the libraries
`pip install numpy pandas matplotlib`

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

