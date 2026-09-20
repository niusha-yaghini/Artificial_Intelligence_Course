Decision Tree Classifier — From Scratch Implementation
Overview

In this project, a Decision Tree classifier was implemented from scratch using Python and NumPy without relying on machine learning libraries for the core algorithm.

The main goal of this implementation was not only to build a working classifier, but also to understand the internal mechanism of Decision Trees, including:

Impurity measurement
Feature splitting
Recursive tree construction
Leaf node generation
Prediction traversal
Model complexity and overfitting
Comparison with Scikit-Learn implementation

After implementing the algorithm manually, the model was evaluated on the Titanic survival prediction dataset and compared with sklearn.tree.DecisionTreeClassifier.

1. What is a Decision Tree?

A Decision Tree is a supervised learning algorithm that makes predictions by learning a sequence of decision rules from data.

Unlike linear models such as Logistic Regression, which learn a mathematical equation:

$$ y = w_1x_1+w_2x_2+...+b $$

Decision Trees learn a set of hierarchical rules:

IF condition:
       go left
ELSE:
       go right

For example:

IF Sex == Female:
        Predict Survived

ELSE:
        Check Fare

The final decisions are stored in leaf nodes.

2. How Decision Trees Learn

The training process consists of finding the best way to split the dataset.

At each node, the algorithm:

Selects a feature
Tries possible thresholds
Splits the dataset into two groups
Measures the quality of the split
Chooses the split that produces the purest child nodes

This process is repeated recursively until a stopping condition is reached.

3. Impurity Measures

A Decision Tree needs a criterion to evaluate how good a split is.

Two common measures were implemented:

Gini Impurity
Entropy
3.1 Gini Impurity

Gini measures how mixed the classes are inside a node.

Formula:

$$ Gini = 1-\sum_{i=1}^{C}p_i^2 $$

where:

\(C\) is the number of classes
\(p_i\) is the probability of class \(i\)

A pure node:

Class 0:
[0,0,0,0]

has:

$$ Gini=0 $$

A mixed node:

[0,1,0,1]

has higher impurity.

3.2 Entropy

Entropy measures the uncertainty of a node.

Formula:

$$ Entropy=-\sum p_i log_2(p_i) $$

A pure node:

[1,1,1]

has:

$$ Entropy=0 $$

Higher entropy means more uncertainty.

4. Implementation From Scratch

The Decision Tree was implemented using the following components:

4.1 Node Class

Each tree node stores:

Feature index
Threshold value
Left child
Right child
Leaf prediction value

Structure:

              Node

        Feature <= Threshold

          /              \

       Left             Right
4.2 Finding the Best Split

For every feature:

Extract possible thresholds
Split the dataset
Calculate weighted impurity
Select the split with minimum impurity

The algorithm searches for:

Best Feature
+
Best Threshold

that separates the classes most effectively.

4.3 Recursive Tree Building

After finding the best split:

The left subset creates the left subtree
The right subset creates the right subtree

The process continues recursively.

Example:

                 Age <= 30

              /             \

          Fare <= 50        Leaf

          /      \

       Leaf      Leaf
4.4 Stopping Criteria

To prevent unlimited growth, several stopping conditions were implemented:

Maximum Depth

Controls tree complexity.

Example:

max_depth=5

means the tree cannot grow deeper than five levels.

Minimum Samples Split

Prevents splitting very small groups.

Example:

min_samples_split=10

means a node needs at least 10 samples before splitting.

5. Testing the Implementation on a Simple Dataset

Before applying the model to a real problem, a small synthetic dataset was used.

Input:

X = [
 [2],
 [3],
 [10],
 [12]
]

Labels:

y = [
0,
0,
1,
1
]

The learned rule was:

Feature <= 3

Left  -> Class 0

Right -> Class 1

Prediction:

[0,0,1,1]

Both Gini and Entropy produced the same result because the dataset was perfectly separable.

6. Titanic Survival Prediction

After validating the implementation, the model was applied to the Titanic dataset.

The preprocessing pipeline included:

Missing value handling
Numerical feature processing
Categorical encoding using One-Hot Encoding

Final feature set:

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

After encoding:

28 features

were used for training.

7. Gini vs Entropy Comparison

Two Decision Trees were trained:

Gini Criterion

Parameters:

criterion = gini
max_depth = 5
min_samples_split = 10

Results:

Metric	Score
Accuracy	0.821
Precision	0.753
Recall	0.797
F1-score	0.775

Confusion Matrix:

[[92 18]
 [14 55]]
Entropy Criterion

Results:

Metric	Score
Accuracy	0.816
Precision	0.765
Recall	0.754
F1-score	0.759

Confusion Matrix:

[[94 16]
 [17 52]]
Analysis

The difference between Gini and Entropy was small.

Both methods successfully learned meaningful decision boundaries.

Gini achieved slightly higher recall and F1-score, while Entropy produced slightly higher precision.

This shows that both impurity measures can perform similarly depending on the dataset.

8. Understanding Overfitting in Decision Trees

Decision Trees are powerful models, but they can easily overfit.

To analyze this behavior, different values of max_depth were tested.

Evaluated depths:

1,2,3,5,7,10,15
Observations
Small Depth

Example:

max_depth=1

Results:

Low training accuracy
Low testing accuracy

The model was too simple.

This is called:

Underfitting
Medium Depth

Example:

max_depth=5

Results:

Good training performance
Best test performance

This represents a good bias-variance tradeoff.

Large Depth

Example:

max_depth=15

Results:

Training accuracy continued increasing
Testing accuracy decreased

The model started memorizing training samples.

This is:

Overfitting
9. Decision Tree Visualization

To better understand the learned rules, the trained tree was visualized.

Example:

Title_Mr <= 0

        |
        |
   FamilySize <= 1.33

        |
        |
     Fare <= 7.29

The visualization showed that the model relied on important Titanic features such as:

Passenger title
Family size
Fare
Fare per person
Age

This demonstrates the interpretability advantage of Decision Trees.

10. Comparison With Scikit-Learn

The custom implementation was compared with:

sklearn.tree.DecisionTreeClassifier

Using identical:

Dataset
Preprocessing
Criterion
Maximum depth
Minimum samples split

Results:

Model	Accuracy	Precision	Recall	F1-score
Custom Decision Tree	0.821	0.753	0.797	0.775
Scikit-Learn Decision Tree	0.821	0.753	0.797	0.775

Confusion Matrix:

[[92 18]
 [14 55]]

Both implementations produced identical predictions.

11. What I Learned

Through this implementation, the following concepts were studied practically:

How Decision Trees select features
How Gini and Entropy measure impurity
How recursive tree construction works
How predictions traverse the tree
Why Decision Trees overfit
How hyperparameters control complexity
How interpretable machine learning models can be
Conclusion

A complete Decision Tree classifier was successfully implemented from scratch.

The custom implementation achieved the same performance as Scikit-Learn's optimized implementation on the Titanic dataset, validating the correctness of the algorithm.

This project provided a complete understanding of Decision Trees from mathematical foundations to practical machine learning application.