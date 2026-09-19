Regression — From Linear Models to Regularization
1. Introduction

Regression is one of the fundamental tasks in supervised learning where the goal is to predict a continuous target variable based on one or more input features.

Unlike classification, where the output belongs to discrete categories, regression models learn a function that maps input variables to a continuous numerical value.

Examples:

Predicting house prices
Estimating stock prices
Forecasting temperature
Predicting demand

The general form of a regression model is:

$$ y=f(X) $$

where:

\(X\) represents input features
\(y\) represents the target value

In this project, regression models were implemented from scratch to understand the mathematical foundations behind common machine learning algorithms.

2. Linear Regression
2.1 Model Definition

Linear Regression assumes that the relationship between input variables and target is approximately linear.

For a single feature:

$$ y=w_0+w_1x $$

For multiple features:

$$ y=w_0+w_1x_1+w_2x_2+...+w_nx_n $$

where:

\(w_0\) is the bias/intercept
\(w_i\) are model coefficients

The model tries to find the optimal weights that minimize prediction error.

2.2 Cost Function

The most common loss function for regression is Mean Squared Error (MSE):

$$ MSE= \frac{1}{m} \sum_{i=1}^{m} (y_i-\hat y_i)^2 $$

where:

\(y_i\) is the true value
\(\hat y_i\) is the predicted value
\(m\) is the number of samples

The goal is:

$$ \min_w MSE $$
2.3 Optimization Using Gradient Descent

Instead of solving the equation analytically, Linear Regression was implemented using Gradient Descent.

Gradient Descent updates model parameters iteratively:

$$ w=w-\alpha\frac{\partial J}{\partial w} $$

where:

\(\alpha\) is learning rate
\(J\) is the loss function

For weights:

$$ dw= \frac{2}{m}X^T(\hat y-y) $$

For bias:

$$ db= \frac{2}{m} \sum(\hat y-y) $$
2.4 Implementation Details

The custom implementation contains:

Initialization
weights = np.zeros(n_features)
bias = 0

Initially, the model has no knowledge about the relationship between inputs and output.

Forward Pass

Prediction:

$$ \hat y=XW+b $$

Implementation:

y_pred = X @ weights + bias
Error Calculation
error = y_pred - y

This represents how far predictions are from real values.

Gradient Calculation
dw = (2/m) * X.T @ error

The gradient determines how weights should change to reduce error.

Parameter Update
weights -= learning_rate * dw
bias -= learning_rate * db
2.5 Evaluation Metrics

Several metrics were used:

MAE

Mean Absolute Error:

$$ MAE= \frac{1}{m} \sum |y-\hat y| $$

Measures average prediction error.

MSE
$$ MSE= \frac1m \sum(y-\hat y)^2 $$

Penalizes large errors more strongly.

RMSE
$$ RMSE=\sqrt{MSE} $$

Same unit as target variable.

R² Score
$$ R^2 = 1-\frac{SS_{res}}{SS_{tot}} $$

Measures how much variance is explained by the model.

2.6 Linear Regression Experiment

Dataset:

California Housing Dataset

Features:

Median income
House age
Average rooms
Population
Location features

Dataset size:

20640 samples
8 features

The data was:

inspected
checked for missing values
standardized before training

Train/Test split:

Train:
16512 samples

Test:
4128 samples

Results:

MAE  = 0.5352
MSE  = 0.5546
RMSE = 0.7447
R2   = 0.5768

The custom implementation achieved results close to Scikit-Learn Linear Regression, validating the correctness of the implementation.

3. Polynomial Regression
3.1 Motivation

Linear Regression assumes:

$$ y \approx wx+b $$

However, many real-world relationships are nonlinear.

Polynomial Regression extends Linear Regression by creating additional polynomial features.

Example:

Original:

$$ x $$

After degree 3 expansion:

$$ [x,x^2,x^3] $$

The model is still linear in parameters:

$$ y=w_0+w_1x+w_2x^2+w_3x^3 $$
3.2 Polynomial Feature Transformation

Using:

PolynomialFeatures(degree=n)

Example:

Degree=2:

$$ X=[x] $$

becomes:

$$ X=[1,x,x^2] $$
3.3 Important Implementation Rule

Polynomial transformation must be fitted only on training data.

Correct:

poly.fit_transform(X_train)

poly.transform(X_test)

Incorrect:

poly.fit_transform(X_test)

because it causes data leakage.

3.4 Bias-Variance Tradeoff

Increasing polynomial degree increases model complexity.

Low degree:

High bias
Underfitting

High degree:

Low bias
High variance
Overfitting

The experiment evaluated:

Degree:
1,2,5,10,15,20

Results showed:

Training error decreased with increasing degree
Test error eventually stopped improving

This demonstrates the bias-variance tradeoff.

3.5 Important Observation

Initially, a surprising behavior appeared:

Test error was lower than training error.

This can happen because:

Random train/test split
Small dataset
Noise distribution differences

It does not always indicate a bug.

4. Regularization

Polynomial models can easily overfit because the number of features increases rapidly.

Regularization controls model complexity by adding a penalty term to the loss function.

General form:

$$ Loss= MSE+\lambda Penalty $$

where:

\(\lambda\) controls regularization strength.

5. Ridge Regression (L2 Regularization)
5.1 Theory

Ridge adds:

$$ \lambda\sum w_i^2 $$

to the loss.

Final objective:

$$ J= MSE+ \lambda \sum w_i^2 $$
5.2 Effect

Ridge:

Reduces large coefficients
Keeps all features
Reduces variance

Weights become smaller:

Before:

[10,8,-5]

After:

[2.1,1.5,-0.8]
5.3 Implementation Difference

Gradient becomes:

$$ dw= MSEGradient+ 2\lambda w $$

Code:

dw += 2 * lambda_ * weights
5.4 Ridge Experiment

Different λ values were tested:

0.001
0.01
0.1
1
10
100

Best result:

lambda = 1

Comparison:

Linear:

Train MSE = 45.65
Test MSE  = 122.98

Ridge:

Train MSE = 53.94
Test MSE  = 81.39

Although training error increased, test performance improved significantly.

This indicates reduced overfitting.

6. Lasso Regression (L1 Regularization)
6.1 Theory

Lasso adds:

$$ \lambda\sum |w_i| $$

Objective:

$$ J= MSE+ \lambda \sum |w_i| $$
6.2 Main Difference From Ridge

Ridge:

$$ w_i \rightarrow small $$

Lasso:

$$ w_i \rightarrow 0 $$

Therefore Lasso can perform feature selection.

6.3 Gradient Update

The derivative of L1 penalty:

$$ \frac{d|w|}{dw}=sign(w) $$

Therefore:

$$ dw= MSEGradient+ \lambda sign(w) $$

Implementation:

dw += lambda_ * np.sign(weights)
6.4 Lasso Experiment

Lambda values:

0
0.001
0.01
0.1
1
50
100
500

Best result:

lambda = 50

Result:

Train MSE = 82.25
Test MSE  = 95.96

Compared with Linear Regression:

Test error decreased:

122.98 → 95.96

showing improved generalization.

6.5 Implementation Note

In this custom Gradient Descent implementation, only a limited number of coefficients became exactly zero.

Professional Lasso implementations usually use:

Coordinate Descent
Proximal Gradient Methods

because L1 regularization has a non-differentiable point at zero.

7. Final Model Comparison
Model	Purpose
Linear Regression	Baseline model
Polynomial Regression	Capture nonlinear relationships
Ridge Regression	Reduce variance
Lasso Regression	Regularization + feature selection

Main observations:

Linear Regression showed overfitting on high-degree polynomial features.
Polynomial expansion improved flexibility but increased variance.
Ridge achieved the best generalization performance.
Lasso reduced complexity and improved generalization compared with unregularized models.
8. Key Lessons Learned

Through this project:

Implemented Linear Regression from scratch.
Understood Gradient Descent optimization.
Learned the effect of polynomial feature expansion.
Observed bias-variance tradeoff experimentally.
Implemented L2 and L1 regularization.
Compared regularized and unregularized models.
Learned why regularization improves generalization.