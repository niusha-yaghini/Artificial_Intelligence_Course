K-Nearest Neighbors (KNN) — From Scratch Implementation
1. Introduction

K-Nearest Neighbors (KNN) is a supervised learning algorithm used for classification and regression tasks.

Unlike parametric models such as Linear Regression or Logistic Regression, KNN does not learn explicit parameters during training.

Instead, it stores the training samples and makes predictions based on similarity between samples.

For this reason, KNN is known as:

Instance-Based Learning
Lazy Learning
2. Problem Definition

The objective is to predict Titanic passenger survival.

Target:

Survived

Classes:

0 → Not Survived

1 → Survived

The same engineered feature set used in Logistic Regression was applied.

Features included:

Numerical:

Age
Fare
FamilySize
FarePerPerson
SibSp
Parch

Categorical:

Sex
Embarked
FamilyCategory
Title
Deck
3. Data Preparation

The preprocessing pipeline:

Feature Engineering

        ↓

Train/Test Split

        ↓

Missing Value Handling

        ↓

One-Hot Encoding

        ↓

Feature Scaling

        ↓

KNN Model
Why Scaling is Important in KNN

KNN relies directly on distance calculations.

The distance between two samples is:

$$ d(x,y)= \sqrt{ \sum_{i=1}^{n}(x_i-y_i)^2 } $$

If features have different scales, features with larger values dominate the distance.

Example:

Age: 0 - 80

Fare: 0 - 500

Fare can dominate the distance calculation.

Therefore StandardScaler was applied before training.

4. KNN Algorithm Theory

For a new sample:

Calculate distance from all training samples.
Select the k nearest samples.
Use majority voting.
Assign the final class.

Mathematically:

$$ \hat y= Mode(y_1,y_2,...,y_k) $$

where:

\(k\) represents the number of neighbors.

5. Distance Function

The implementation uses Euclidean Distance:

$$ d= \sqrt{ \sum(x_i-y_i)^2 } $$

For each test sample:

Distance to all training samples is calculated.
The nearest neighbors are selected.
Their labels determine the prediction.
6. Implementation Details

The custom KNN implementation contains:

Fit

Unlike optimization-based models, KNN does not learn weights.

The fit function only stores:

Training Features

Training Labels
Predict

For each test sample:

Calculate distances.
Sort distances.
Select k nearest neighbors.
Perform majority voting.
7. Selecting the Best K Value

The value of k controls the bias-variance tradeoff.

Small k

Example:

k = 1

Characteristics:

High flexibility
Sensitive to noise
High variance
Possible overfitting

The experiment showed:

Train Accuracy ≈ 98%
Test Accuracy ≈ 73%
Large k

Example:

k = 50

Characteristics:

More stable predictions
Lower variance
Higher bias

Very large k values may cause underfitting.

8. K Value Experiment Results

Different values of k were tested:

1,3,5,7,9,15,25,50

Results:

k	Test Accuracy	F1-score
1	0.732	0.631
5	0.793	0.699
15	0.804	0.729
25	0.821	0.758
50	0.810	0.742

The best performance was achieved with:

k = 25
9. Final Model Evaluation

Using:

k = 25

The model achieved:

Metric	Score
Accuracy	0.821
Precision	0.794
Recall	0.725
F1-score	0.758

Confusion Matrix:

[[97 13]

 [19 50]]
10. Comparison With Scikit-Learn

The custom implementation was compared with:

KNeighborsClassifier

Both models used:

k = 25

Euclidean Distance

Uniform Voting

Results:

Model	Accuracy	Precision	Recall	F1
Custom KNN	0.821	0.794	0.725	0.758
Scikit-Learn KNN	0.821	0.794	0.725	0.758

The identical results validate the correctness of the custom implementation.

11. Key Learning Points

Through this project:

Implemented KNN from scratch.
Learned instance-based learning.
Understood distance-based classification.
Learned the importance of feature scaling.
Studied bias-variance tradeoff using k.
Analyzed overfitting and underfitting.
Compared custom implementation with Scikit-Learn.
12. Limitations of KNN

Although KNN is simple and intuitive, it has limitations:

Prediction can be computationally expensive.
Performance decreases in high-dimensional spaces.
Sensitive to feature scaling.
Sensitive to irrelevant features.