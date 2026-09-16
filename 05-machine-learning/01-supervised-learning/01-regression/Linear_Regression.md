# Linear Regression From Scratch

## Overview

Linear Regression is one of the fundamental supervised learning algorithms used for predicting a continuous target variable.

The main idea is to learn a linear relationship between input features and the target variable by estimating a set of parameters (weights and bias).

In this project, Linear Regression was implemented from scratch using Gradient Descent and validated against Scikit-Learn's implementation.

---

# Problem Definition

Given a dataset containing input features:

$X = [x_1,x_2,...,x_n]$

the goal is to predict a continuous output:

$y$

The model tries to learn a function:

$\hat{y}=f(X)$

where:

$\hat{y}$

is the predicted value.

---

# Dataset

## California Housing Dataset

The California Housing dataset from Scikit-Learn was used for this experiment.

Dataset characteristics:

- Samples: 20640
- Features: 8
- Target: Median House Value

Features:

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude

Target:

- MedHouseVal

The dataset contains numerical features and does not require categorical encoding.

---

# Exploratory Data Analysis

Before training the model, several analyses were performed:

## Missing Values

The dataset was checked for missing values.

Result:

- No missing values were found.

---

## Correlation Analysis

The correlation between features and target was analyzed.

Main observations:

- `MedInc` showed the strongest positive correlation with house value.
- Some features such as `AveRooms` and `AveBedrms` showed high correlation with each other, indicating possible multicollinearity.
- Geographic features (`Latitude`, `Longitude`) also showed relationships with house prices.

---

# Mathematical Formulation

For a single feature:

$\hat{y}=wx+b$

For multiple features:

$\hat{y}
=
w_1x_1+w_2x_2+...+w_nx_n+b$


In matrix form:

$\hat{y}=XW+b$


where:

- \(X\) represents input features
- \(W\) represents learnable weights
- \(b\) represents bias term

---

# Cost Function

To measure the difference between predictions and actual values, Mean Squared Error (MSE) was used.

$MSE=
\frac{1}{m}
\sum_{i=1}^{m}
(y_i-\hat{y_i})^2$


The goal of training is to minimize this loss function.

---

# Gradient Descent Optimization

Instead of solving Linear Regression analytically, Gradient Descent was implemented.

The idea is to iteratively update model parameters in the direction that minimizes the loss.


Weight update:

$W=W-\alpha\frac{\partial J}{\partial W}$


Bias update:

$b=b-\alpha\frac{\partial J}{\partial b}$


where:

- \(\alpha\) is the learning rate
- \(J\) is the cost function

---

# Implementation Details

A custom class:

```python
LinearRegressionGD