** All datasets are using locally, but the link for downloading them are gonna be provided in each case needed.

# Machine Learning

This section covers the fundamental concepts, algorithms, and practical implementations of classical Machine Learning.

The goal of this section is not only to use existing libraries, but also to understand the mathematical foundations and internal mechanisms of ML algorithms by implementing them from scratch and comparing them with standard libraries such as Scikit-learn.

The learning approach follows this workflow:

1. Understand the mathematical intuition behind each algorithm.
2. Implement the algorithm from scratch.
3. Evaluate the model on real-world datasets.
4. Analyze strengths, limitations, and behavior.
5. Compare the implementation with established ML libraries.

---

# Learning Structure

## 00 - Data Preprocessing and Exploratory Data Analysis

Before building machine learning models, understanding and preparing the data is essential.

This section covers:

- Dataset structure and characteristics
- Feature and target variables
- Handling missing values
- Data cleaning
- Outlier detection
- Feature scaling
- Encoding categorical variables
- Train/Test splitting
- Exploratory Data Analysis (EDA)

The objective is to build a strong foundation for reliable ML pipelines.

---

# 01 - Supervised Learning

Supervised learning focuses on learning a mapping between input features and known target labels.

This section is divided into regression and classification problems.

---

## Regression

Regression models predict continuous numerical values.

Implemented topics:

- Linear Regression
- Polynomial Regression
- Regularization techniques

Main concepts:

- Linear models
- Loss functions
- Gradient Descent
- Overfitting
- Bias-Variance Tradeoff
- Ridge and Lasso regularization

---

## Classification

Classification models predict discrete class labels.

Implemented topics:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Trees
- Random Forest
- Gradient Boosting

Main concepts:

- Decision boundaries
- Probabilistic classification
- Tree-based learning
- Ensemble methods

---

# 02 - Model Evaluation

A machine learning model is not only defined by its predictions, but also by how reliably it performs on unseen data.

This section covers:

## Regression Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

## Model Validation

- Cross Validation
- Hyperparameter evaluation
- Bias-Variance analysis
- Overfitting and Underfitting

---

# 03 - Feature Engineering

Feature engineering focuses on improving machine learning performance by creating and selecting informative features.

Topics include:

- Feature creation
- Feature transformation
- Feature selection
- Dimensionality-related techniques

The goal is to understand how data representation affects model performance.

---

# 04 - Unsupervised Learning

Unsupervised learning focuses on discovering hidden patterns in data without labeled outputs.

Implemented topics:

## Clustering

- K-Means
- DBSCAN
- Hierarchical Clustering

## Dimensionality Reduction

- Principal Component Analysis (PCA)

Main concepts:

- Similarity and distance measures
- Cluster formation
- Data representation
- Feature space transformation

---

# Implementation Philosophy

Each algorithm follows the same development pipeline:

Mathematical Understanding
↓
From-Scratch Implementation
↓
Experimentation
↓
Performance Analysis
↓
Library Comparison



The from-scratch implementations are developed to understand the internal behavior of algorithms, while library implementations demonstrate practical usage in real-world projects.

---

# Tools and Libraries

Main tools used in this section:

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Dataset Usage

Real-world and benchmark datasets are used throughout this section.

Datasets are organized separately in the main project directory:

datasets/



Sources include:

- UCI Machine Learning Repository
- Kaggle
- OpenML
- Scikit-learn datasets

---

# Future Extensions

Planned extensions include:

- Support Vector Machines and Kernel Methods
- Hyperparameter Optimization
- Probabilistic Machine Learning
- Reinforcement Learning
- Neural Networks and Deep Learning


--------------------------------------

So far, we have primarily focused on this question:

"How do we find a good solution to an optimization problem?"

For instance:

The best TSP route?
The best parameter vector?
The best expression?

However, in machine learning, the question changes:

"How do we learn a model from data that can make predictions on new data?"

--------------------------------------

each notebook has this structure:

1. Problem Introduction
2. Dataset Description
3. Mathematical Background
4. From Scratch Implementation
5. Training
6. Evaluation
7. Visualization
8. sklearn Implementation
9. Scratch vs sklearn Comparison
10. Conclusion

------------------------------------

Project advancement process:

Data
 ↓
Model
 ↓
Measure
 ↓
Improve Features
 ↓
Explore Unlabeled Data

------------------------------------

