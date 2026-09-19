Logistic Regression — From Scratch Implementation
1. Introduction

Logistic Regression is one of the fundamental algorithms in supervised learning used for binary classification problems.

Unlike Linear Regression, which predicts continuous values, Logistic Regression estimates the probability that a sample belongs to a specific class.

Examples:

Spam detection
Disease diagnosis
Customer churn prediction
Survival prediction

In this project, Logistic Regression was implemented from scratch using:

Sigmoid activation function
Binary Cross Entropy loss
Gradient Descent optimization

The implementation was evaluated and compared with Scikit-Learn's Logistic Regression.

2. Problem Definition

The goal is to predict whether a Titanic passenger survived.

Target:

Survived

where:

0 → Not Survived

1 → Survived

Input features include engineered variables:

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
3. Data Preprocessing Pipeline

Before training, the dataset was processed through the following pipeline:

Feature Engineered Dataset

        ↓

Train/Test Split

        ↓

Missing Value Handling

        ↓

Feature Scaling

        ↓

One-Hot Encoding

        ↓

Logistic Regression
3.1 Missing Value Handling

Numerical features:

SimpleImputer(strategy="median")

Categorical features:

SimpleImputer(strategy="most_frequent")

Median was used for numerical features because it is more robust against outliers.

3.2 Feature Scaling

Numerical features were standardized using:

$$ z= \frac{x-\mu}{\sigma} $$

Using:

StandardScaler()

Scaling is important because Gradient Descent depends on feature magnitude.

3.3 One-Hot Encoding

Categorical features were converted into numerical representations.

Example:

Before:

Sex

male
female

After:

Sex_male
Sex_female

Using:

OneHotEncoder()
4. Logistic Regression Theory
4.1 Linear Combination

Similar to Linear Regression:

$$ z=XW+b $$

where:

X = input features
W = weights
b = bias

However, the output is not directly used.

4.2 Sigmoid Function

To convert the output into probability:

$$ \sigma(z)= \frac{1}{1+e^{-z}} $$

The sigmoid function maps any value into:

$$ 0 \leq p \leq 1 $$

Example:

z = 3

Probability = 0.95
4.3 Binary Cross Entropy Loss

Because this is a binary classification problem, Mean Squared Error is not suitable.

The model minimizes:

$$ J= -\frac1m \sum [ y\log(p) + (1-y)\log(1-p) ] $$

where:

y = true label
p = predicted probability
5. Gradient Descent Optimization

The gradients are:

Weights:

$$ dw= \frac1m X^T(p-y) $$

Bias:

$$ db= \frac1m \sum(p-y) $$

Parameters update:

$$ W=W-\alpha dw $$ $$ b=b-\alpha db $$

where:

\(\alpha\) is learning rate.

6. Implementation Details

The custom implementation contains:

Sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))
Training

During each epoch:

Calculate linear output
$$ z=XW+b $$
Convert to probability
$$ p=\sigma(z) $$
Calculate gradients
Update parameters
Store loss history
Prediction

Two prediction modes were implemented:

Probability prediction
predict_proba()

Example:

0.82
0.15
0.63
Class prediction
predict()

Using threshold:

$$ p \geq threshold $$

Default:

threshold=0.5
7. Model Training Results

Training configuration:

Learning rate = 0.01

Epochs = 5000

The training loss decreased:

Initial loss:

0.693

Final loss:

0.416

This indicates successful convergence of Gradient Descent.

8. Model Evaluation

The custom Logistic Regression achieved:

Accuracy = 82.68%

Confusion Matrix:

              Predicted

              0     1

Actual 0      98    12

Actual 1      19    50

Classification metrics:

Class	Precision	Recall	F1-score
0	0.84	0.89	0.86
1	0.81	0.72	0.76
9. Comparison With Scikit-Learn

Two models were compared:

Model	Accuracy	Precision	Recall	F1
Custom Logistic Regression	0.827	0.806	0.725	0.763
Scikit-Learn Logistic Regression	0.827	0.788	0.754	0.770

Both models achieved almost identical accuracy.

This confirms that the custom Gradient Descent implementation correctly learned the classification boundary.

10. Threshold Analysis

The default classification threshold:

0.5

was compared with different thresholds.

Best result:

Threshold = 0.4

Results:

Metric	0.5	0.4
Accuracy	0.827	0.832
Recall	0.725	0.768
F1-score	0.763	0.779

Changing the threshold improved the balance between precision and recall.

11. Key Learning Points

Through this project:

Implemented Logistic Regression from scratch.
Learned probability-based classification.
Implemented Sigmoid activation.
Implemented Binary Cross Entropy loss.
Applied Gradient Descent optimization.
Learned classification evaluation metrics.
Compared custom implementation with Scikit-Learn.
Understood the effect of decision thresholds.