Random Forest Classifier — From Scratch Implementation
Overview

In this project, a Random Forest classifier was implemented from scratch using Python and NumPy without using machine learning libraries for the core algorithm.

The main goal was to understand the internal mechanism of Random Forest, including:

Ensemble Learning
Bootstrap Sampling
Decision Tree Aggregation
Feature Randomness
Majority Voting
Hyperparameter effects
Comparison with Scikit-Learn implementation

The custom implementation was evaluated on the Titanic survival prediction dataset and compared with sklearn.ensemble.RandomForestClassifier.

1. What is Random Forest?

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to create a stronger and more stable model.

A single Decision Tree can easily overfit because it learns very specific patterns from the training data.

Random Forest reduces this problem by:

Training many different Decision Trees
Using random subsets of data
Using random subsets of features
Combining their predictions using voting

The main idea:

Multiple Weakly Correlated Trees
              |
              |
        Majority Voting
              |
              |
        Final Prediction
2. Why Not Use Only One Decision Tree?

A Decision Tree has several advantages:

Easy interpretation
Handles nonlinear relationships
Requires little preprocessing

However, it has a major limitation:

High Variance

Small changes in training data can create a completely different tree.

Example:

Dataset A

        Age
         |
      Fare


Dataset B

        Sex
         |
     FamilySize

Both trees may perform well on training data but generalize differently.

Random Forest solves this by averaging many different trees.

3. Main Concepts Behind Random Forest

Random Forest consists of two main sources of randomness:

3.1 Bootstrap Sampling

Each Decision Tree is trained on a different random sample of the original dataset.

Sampling is performed with replacement.

Example:

Original dataset:

1000 samples

Tree 1:

Sample:
1, 5, 20, 20, 300,...

Tree 2:

Sample:
4, 8, 8, 500,...

Because each tree sees a slightly different dataset, the trees learn different patterns.

3.2 Feature Randomness

In a normal Decision Tree:

At each split:

All features are considered

Example:

Feature 1
Feature 2
Feature 3
...
Feature 28

The best split is selected.

In Random Forest:

Only a random subset of features is considered.

Example:

28 total features

Random subset:

Feature 3
Feature 7
Feature 15
Feature 22
Feature 25

This reduces similarity between trees and improves generalization.

4. Random Forest Architecture

The implemented model consists of:

                Dataset

                   |
          Bootstrap Sampling

     -----------------------------

     Tree 1      Tree 2      Tree 3

       |           |           |

 Prediction  Prediction  Prediction

     -----------------------------

             Majority Voting

                   |

              Final Output
5. Implementation From Scratch

The implementation contains two main components:

5.1 Decision Tree for Random Forest

A separate Decision Tree class was created specifically for Random Forest.

It supports:

Gini impurity
Entropy
Maximum depth control
Minimum samples split
Random feature selection

Each tree contains:

Node splitting
Recursive tree growth
Leaf node prediction
Tree traversal during prediction
5.2 Random Forest Class

The Random Forest class manages multiple Decision Trees.

Main steps:

Training

For each estimator:

Create bootstrap sample
Initialize a Decision Tree
Train the tree
Store the trained tree

Pseudo-code:

for each tree:

    Create bootstrap dataset

    Train Decision Tree

    Save tree
Prediction

During prediction:

Every tree predicts the class
Predictions are collected
Majority voting determines final output

Example:

Tree 1 → Survived

Tree 2 → Survived

Tree 3 → Not Survived

Tree 4 → Survived


Final Prediction → Survived
6. Dataset and Preprocessing

The Titanic dataset was used for evaluation.

The preprocessing pipeline included:

Handling missing values
Numerical feature processing
Categorical feature encoding
Feature transformation

Final processed dataset:

Training samples: 712

Testing samples: 179

Number of Features: 28

Features included:

Age
Fare
FamilySize
FarePerPerson
SibSp
Parch
Sex
Embarked
FamilyCategory
Title
Deck
7. Initial Random Forest Evaluation

The first experiment used:

n_estimators = 10

criterion = gini

max_depth = 5

min_samples_split = 10

max_features = 5

Results:

Model	Accuracy	Precision	Recall	F1-score
Random Forest	0.7877	0.7627	0.6522	0.7031

The initial Random Forest did not outperform the previous Decision Tree model.

This is expected because:

The number of trees was small
The dataset size was limited
Hyperparameters were not optimized yet

Further experiments were performed to analyze the effect of Random Forest parameters.

8. Effect of Number of Trees

The number of estimators was changed:

1
5
10
25
50
100

Results:

Number of Trees	Accuracy	F1-score
1	0.7486	0.6980
5	0.7989	0.7273
10	0.7821	0.7023
25	0.8045	0.7328
50	0.8268	0.7704
100	0.8268	0.7669
Analysis

Increasing the number of trees improved performance by reducing variance.

The model improved from:

Single Tree:

Accuracy = 74.86%

to:

50 Trees:

Accuracy = 82.68%

After approximately 50 trees, performance became stable.

Adding more trees provided limited improvement while increasing computational cost.

9. Effect of Maximum Features

The effect of feature randomness was evaluated using:

max_features = 5

max_features = 14

max_features = None

Results:

Max Features	Accuracy	F1-score
5	0.8268	0.7669
14	0.8156	0.7591
None	0.8212	0.7746
Analysis

Using fewer features increased randomness between trees and improved accuracy.

The best accuracy was achieved with:

max_features = 5

However, using all features achieved higher recall and F1-score.

This demonstrates the trade-off between:

Tree diversity
Individual tree strength
10. Comparison With Scikit-Learn Random Forest

The final custom implementation was compared with:

sklearn.ensemble.RandomForestClassifier

Using the same:

Dataset
Preprocessing
Number of estimators
Tree depth
Feature selection

Parameters:

n_estimators = 50

criterion = gini

max_depth = 5

min_samples_split = 10

max_features = 5

Results:

Model	Accuracy	Precision	Recall	F1-score
Custom Random Forest	0.8324	0.8095	0.7391	0.7727
Scikit-Learn Random Forest	0.8212	0.7761	0.7536	0.7647
Analysis

The custom implementation achieved slightly higher:

Accuracy
Precision
F1-score

while Scikit-Learn achieved slightly higher recall.

The small difference is expected because Random Forest contains random components:

Bootstrap sampling
Random feature selection

Therefore, two implementations may produce slightly different decision boundaries.

The close performance validates that the core concepts were correctly implemented:

Multiple Decision Trees
Bootstrap Sampling
Feature Randomness
Majority Voting
11. Visualization

The final comparison was visualized using a metric comparison chart.

The visualization shows that both implementations achieved very similar performance:

Custom Random Forest ≈ Scikit-Learn Random Forest

This confirms the correctness of the custom implementation.

12. What I Learned

Through this project, the following concepts were studied:

Ensemble Learning
Bagging
Bootstrap Sampling
Feature Randomness
Variance Reduction
Majority Voting
Hyperparameter tuning
Model comparison

The implementation provided a deeper understanding of how Random Forest works internally rather than treating it as a black-box algorithm.

Conclusion

A complete Random Forest classifier was successfully implemented from scratch.

The model achieved performance comparable to Scikit-Learn's optimized implementation on the Titanic dataset.

This project demonstrated how combining multiple diverse Decision Trees can improve model stability and reduce overfitting compared with using a single Decision Tree.