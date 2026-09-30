# Dimensionality Reduction

## Overview

Dimensionality Reduction is an unsupervised learning technique used to transform high-dimensional data into a lower-dimensional representation while preserving important information.

High-dimensional datasets often contain redundant features, noise, and complex structures that are difficult to visualize or process.

Dimensionality reduction helps with:

- Data visualization
- Reducing computational cost
- Removing noise
- Avoiding the curse of dimensionality
- Improving model performance in some cases

In this section, three dimensionality reduction techniques were studied and implemented:

1. Principal Component Analysis (PCA)
2. Singular Value Decomposition (SVD)
3. t-Distributed Stochastic Neighbor Embedding (t-SNE)

All methods were evaluated using the Iris dataset.

---

# 1. Principal Component Analysis (PCA)

## Concept

Principal Component Analysis is a linear dimensionality reduction method that transforms original features into a new set of features called Principal Components.

The goal of PCA is:

> Find directions that maximize the variance of the data.

Each Principal Component is a linear combination of the original features:

\[
PC_i = w_1X_1+w_2X_2+...+w_nX_n
\]

The first components contain the largest amount of information.

---

# PCA Algorithm

## Step 1 — Centering Data

First, the mean of each feature is removed:

\[
X=X-\mu
\]

This moves the data distribution around zero.

---

## Step 2 — Covariance Matrix

The covariance matrix is calculated:

\[
C=\frac{1}{n-1}X^TX
\]

This matrix describes relationships between features.

---

## Step 3 — Eigen Decomposition

Eigenvalues and eigenvectors are calculated:

\[
Cv=\lambda v
\]

Where:

- Eigenvectors represent component directions.
- Eigenvalues represent explained variance.

---

## Step 4 — Component Selection

Components are sorted based on eigenvalues.

The components with the highest variance are selected.

---

# PCA Implementation

A custom PCA implementation was created with:

- Feature centering
- Covariance matrix calculation
- Eigen decomposition
- Component selection
- Explained variance calculation
- Data transformation
- Data reconstruction

The implementation was compared with:

```python
sklearn.decomposition.PCA

PCA Results on Iris Dataset
Original dataset:
150 samples
4 features

After PCA:
150 samples
2 components

The explained variance ratio was:
Component	Explained Variance
PC1	72.96%
PC2	22.85%


Total preserved information:
95.81%

This means PCA reduced the dimensionality by half while preserving almost all important information.
PCA Visualization Analysis
The PCA projection showed:
- Setosa was clearly separated from other classes.
- Versicolor and Virginica remained partially overlapping.
This happens because these two classes have similar feature distributions.
2. Singular Value Decomposition (SVD)
Concept
Singular Value Decomposition is a matrix factorization technique that decomposes a matrix into three components:
\[
X=U\Sigma V^T
\]
Where:
- U contains left singular vectors.
- Σ contains singular values.
- Vᵀ contains right singular vectors.
SVD is widely used in:
- Dimensionality reduction
- Recommendation systems
- Data compression
- Matrix analysis
Relationship Between SVD and PCA
PCA and SVD are mathematically connected.
PCA uses:
Covariance Matrix
        ↓
Eigen Decomposition

SVD uses:
Data Matrix
        ↓
Matrix Factorization

The principal directions found by PCA are equivalent to the right singular vectors of SVD.
SVD Algorithm
Step 1
Compute:
\[
X^TX
\]
Step 2
Perform eigen decomposition:
\[
X^TXv=\lambda v
\]
Step 3
Calculate singular values:
\[
\sigma=\sqrt{\lambda}
\]
Step 4
Select the largest singular values and corresponding vectors.
SVD Implementation
A custom SVD implementation was created with:
- Eigen decomposition
- Singular value extraction
- Component selection
- Data projection
- Reconstruction
The implementation was compared with:
numpy.linalg.svd


SVD Results on Iris Dataset
The original dataset:
150 samples
4 features

was transformed into:
150 samples
2 components

The first singular values were:
Component	Singular Value
1	20.92
2	11.70


The reconstruction error:
MSE = 0.0418

shows that the reduced representation preserved most of the important structure.
SVD Visualization Analysis
The SVD projection produced a visualization very similar to PCA.
Observed:
- Clear separation of Setosa.
- Overlap between Versicolor and Virginica.
This confirms the mathematical relationship between PCA and SVD.
3. t-SNE
Concept
t-SNE is a nonlinear dimensionality reduction technique mainly designed for visualization.
Unlike PCA, which maximizes variance, t-SNE tries to preserve local neighborhood relationships.
The main idea:
Similar points in high-dimensional space should remain close in low-dimensional space.

t-SNE Algorithm
Step 1 — Similarity Calculation
For every pair of points, t-SNE calculates similarity probabilities:
\[
p_{ij}
\]
These probabilities represent how close samples are in the original space.
Step 2 — Low-dimensional Mapping
A lower-dimensional representation is created:
\[
q_{ij}
\]
Step 3 — Optimization
The algorithm minimizes the difference between both distributions:
\[
KL(P||Q)
\]
using gradient descent.
Important Parameters
Perplexity
The most important t-SNE parameter.
It controls the effective number of neighbors.
Low perplexity:
- Focuses on local structure.
- Can create fragmented clusters.
High perplexity:
- Considers more neighbors.
- Preserves more global structure.
Common values:
5 - 50

t-SNE Implementation
Unlike PCA and SVD, a complete implementation of t-SNE from scratch requires:
- Probability distributions
- KL divergence
- Gradient descent optimization
- Student-t distribution
Therefore, sklearn implementation was used:
sklearn.manifold.TSNE


The main parameters were evaluated and visualized.
t-SNE Results on Iris Dataset
The dataset was reduced:
4 dimensions

to:
2 dimensions

The visualization showed:
- Strong separation of Setosa.
- Partial overlap between Versicolor and Virginica.
Perplexity Analysis
Different perplexity values were tested:
Perplexity	Observation
5	Focused heavily on local structures and produced scattered groups
20	Provided balanced representation
30	Produced clear cluster separation
50	Captured more global structure but reduced some separation


The experiment showed that t-SNE output strongly depends on parameter selection.
Comparison of Dimensionality Reduction Methods
Method	Type	Main Goal	Strength	Limitation
PCA	Linear	Maximum variance preservation	Fast and interpretable	Cannot capture nonlinear structures
SVD	Linear	Matrix factorization	Efficient decomposition	Same limitations as linear methods
t-SNE	Nonlinear	Local neighborhood preservation	Excellent visualization	Cannot preserve global distances


Final Conclusion
The experiments demonstrated different approaches to dimensionality reduction.
PCA
Successfully reduced Iris from 4 dimensions to 2 dimensions while preserving:
95.81% variance

It provided an interpretable linear transformation.
SVD
Produced results equivalent to PCA and demonstrated how matrix decomposition can be used for dimensionality reduction.
t-SNE
Provided a nonlinear visualization where cluster structures became more apparent, especially for separating Setosa.
However, t-SNE should mainly be used for visualization because distances in the embedding space are not directly meaningful.
Overall:
- PCA is suitable for general dimensionality reduction.
- SVD is useful for mathematical decomposition and compression.
- t-SNE is best suited for exploring complex high-dimensional datasets visually.