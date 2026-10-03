# Support Vector Machines and Kernels

This section explores Support Vector Machines (SVM), one of the fundamental algorithms in machine learning for classification and regression tasks.

The goal of this section is to understand the mathematical intuition behind SVM, implement a linear SVM from scratch, and study how kernel methods allow SVM to solve non-linear problems.

---

# Overview

Support Vector Machine is a supervised learning algorithm that finds an optimal decision boundary between classes by maximizing the margin between them.

The main idea:

- Find the best separating hyperplane.
- Maximize the distance between the hyperplane and the closest samples.
- Use support vectors to define the decision boundary.

SVM is especially effective for:

- Small and medium-sized datasets
- High-dimensional feature spaces
- Problems where clear class boundaries exist

---

# Structure

06-support-vector-machines-and-kernels/
├── 01_svm_basics.ipynb
├── 02_svm_from_scratch.ipynb
└── 03_kernel_methods.ipynb

---

# 01 - SVM Basics

## Concepts Covered

### Hyperplane

SVM creates a decision boundary:

\[
w^Tx+b=0
\]

where:

- `w` represents the weight vector
- `b` represents the bias term

The sign of the output determines the predicted class.

---

## Margin Maximization

Instead of only finding a separating line, SVM tries to maximize the margin:

\[
Margin=\frac{2}{||w||}
\]

A larger margin usually produces a more robust classifier.

---

## Support Vectors

Support vectors are the closest samples to the decision boundary.

They are the most important points because they determine the position of the hyperplane.

Changing or removing distant samples usually has little effect on the model.

---

## Soft Margin and Parameter C

Real-world data is usually not perfectly separable.

SVM introduces Soft Margin, allowing some classification errors.

The parameter `C` controls the trade-off:

- Small `C`
    - Larger margin
    - More tolerance for errors
    - Simpler model

- Large `C`
    - Smaller margin
    - Fewer classification errors
    - More complex model

---

## Visualization

The decision boundary and margins were visualized using:

- Decision boundary (`f(x)=0`)
- Margin boundaries (`f(x)=±1`)
- Support vectors

---

# 02 - Linear SVM From Scratch

In this notebook, a Linear SVM was implemented without using Scikit-Learn.

The implementation includes:

- Weight initialization
- Gradient Descent optimization
- Hinge Loss
- L2 Regularization
- Decision Function
- Prediction
- Support Vector detection

---

# Hinge Loss

SVM uses hinge loss to penalize samples inside the margin:

\[
Loss=max(0,1-y(w^Tx+b))
\]

If a sample is correctly classified and outside the margin:

\[
Loss=0
\]

Otherwise, the model updates its parameters.

---

# Optimization

The objective function combines:

### Margin maximization

\[
\frac{1}{2}||w||^2
\]

### Classification error penalty

\[
C\sum\xi_i
\]

Gradient descent is used to update:

- weights (`w`)
- bias (`b`)

---

# Custom SVM vs Scikit-Learn SVM

The custom implementation was compared with:


sklearn.svm.SVC(kernel="linear")

Results:

- Both models achieved similar classification accuracy.
- Decision boundaries were visually similar.
- The custom model reproduced the main behavior of Linear SVM.

Differences appeared because:

- Scikit-Learn uses an optimized SVM solver.
- The custom model uses gradient descent approximation.

Therefore:

- Weight values are not identical.
- The detected support vectors can differ.

---

# 03 - Kernel Methods

Linear SVM can only create straight decision boundaries.

For non-linear datasets, Kernel Methods allow SVM to work in higher-dimensional feature spaces.

---

# Kernel Trick

Instead of explicitly creating new features, SVM uses a kernel function:

\[
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)
\]

This allows the model to operate in a higher-dimensional space without explicitly computing the transformation.

---

# Linear Kernel

The standard linear SVM:

\[
K(x_i,x_j)=x_i^Tx_j
\]

Suitable for linearly separable datasets.

---

# Polynomial Kernel

Polynomial Kernel creates curved decision boundaries.

Formula:

\[
K(x_i,x_j)=(x_i^Tx_j+c)^d
\]

Important parameter:

`degree`

Higher degree:

- More complex boundary
- Higher risk of overfitting

---

# RBF Kernel

Radial Basis Function is one of the most commonly used kernels.

Formula:

\[
K(x_i,x_j)=e^{-\gamma ||x_i-x_j||^2}
\]

The parameter `gamma` controls the influence of individual samples.

### Small gamma

- Smooth boundaries
- Simpler model

### Large gamma

- More complex boundaries
- Higher risk of overfitting

---

# Kernel Comparison

Different kernels were tested on a non-linear dataset (`make_moons`):

## Linear SVM

- Produced a straight boundary
- Could not fully capture the data structure

## Polynomial Kernel

- Created curved boundaries
- Improved performance

## RBF Kernel

- Produced the most flexible boundary
- Achieved the best performance on the non-linear dataset

---

# Key Learnings

Through this section, the following concepts were implemented and analyzed:

- Maximum Margin Classification
- Hyperplanes
- Support Vectors
- Soft Margin SVM
- Effect of parameter `C`
- Hinge Loss
- Gradient Descent SVM implementation
- Kernel Trick
- Polynomial Kernel
- RBF Kernel
- Effect of `gamma`

---

# Final Conclusion

Support Vector Machines provide a powerful framework for classification by maximizing the margin between classes.

Although modern large-scale applications often rely on deep learning models, SVM remains an important algorithm for:

- Small datasets
- High-dimensional problems
- Strong baseline comparisons
- Problems with structured feature spaces

This section demonstrates both the theoretical foundation and practical implementation of SVM, from a basic linear classifier to advanced kernel-based decision boundaries.