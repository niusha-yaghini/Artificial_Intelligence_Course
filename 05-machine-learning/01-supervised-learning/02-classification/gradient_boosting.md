# Gradient Boosting Classification

## Overview

Gradient Boosting is an ensemble learning algorithm that combines multiple weak learners (usually decision trees) to create a stronger predictive model.

Unlike Random Forest, where trees are built independently and their predictions are combined using voting, Gradient Boosting builds trees sequentially. Each new tree attempts to correct the errors made by previous trees.

The main idea is:


Initial Model
      |
      v
Calculate Errors
      |
      v
Train New Tree on Errors
      |
      v
Update Model
      |
      v
Repeat

Gradient Boosting is widely used in machine learning competitions and real-world applications because it can capture complex relationships between features while maintaining good generalization performance.

---

# 1. How Gradient Boosting Works

Gradient Boosting builds an additive model:

\[
F(x)=F_0(x)+\eta h_1(x)+\eta h_2(x)+...+\eta h_m(x)
\]

Where:

- \(F_0(x)\) is the initial prediction
- \(h_m(x)\) represents the prediction of each new decision tree
- \(\eta\) is the learning rate
- \(m\) is the number of trees

Each tree contributes a small correction to the previous model.

---

# 2. Gradient Boosting vs Random Forest

Although both methods use multiple decision trees, their learning strategies are different.

| Feature | Random Forest | Gradient Boosting |
|---|---|---|
| Ensemble Type | Bagging | Boosting |
| Tree Training | Parallel | Sequential |
| Main Goal | Reduce variance | Reduce bias |
| Tree Dependency | Independent | Depends on previous trees |
| Randomness | High | Usually low |
| Learning Process | Voting | Error correction |

Random Forest creates many independent trees and combines their predictions.

Gradient Boosting creates trees one after another, where every tree tries to improve the previous model.

---

# 3. Gradient Boosting for Classification

For regression problems, Gradient Boosting directly learns numerical residuals.

For classification problems, the process is slightly different.

The model does not directly predict class labels. Instead, it learns a score (logit), which is converted into probability using the sigmoid function.

The workflow is:


Features
   |
   v
Gradient Boosting Trees
   |
   v
Raw Prediction Score
   |
   v
Sigmoid Function
   |
   v
Probability
   |
   v
Class Prediction

For binary classification:


Probability >= 0.5  -> Class 1
Probability < 0.5   -> Class 0

---

# 4. Implementation Approach

In this project, Gradient Boosting was implemented in two stages.

## Stage 1 — Regression Tree Implementation

The fundamental component of Gradient Boosting is a regression tree.

Unlike classification trees:

Classification Tree:


Leaf -> Class label

Regression Tree:


Leaf -> Mean numerical value

Example:


Values:
[10,20,30]
Prediction:
20

---

# 5. Regression Tree Structure

The regression tree contains:

- Nodes
- Feature selection
- Threshold splitting
- Leaf predictions

The splitting criterion is Mean Squared Error (MSE).

The goal is finding the split that minimizes:

\[
MSE=\frac{1}{n}\sum(y-\bar{y})^2
\]

The best split is the one that creates groups with the lowest internal variance.

---

# 6. Regression Tree Implementation

The implemented regression tree contains:

## Node

Each node stores:

- Selected feature
- Split threshold
- Left child
- Right child
- Leaf value


## Tree Growth

The tree is built recursively:


Current Node
  |

Find Best Split
  |

Left        Right
  |

Repeat

Stopping conditions:

- Maximum depth reached
- Not enough samples for splitting
- No valid split found

---

# 7. Gradient Boosting Implementation

After creating the regression tree, Gradient Boosting was implemented using:

## Initial Prediction

The initial model predicts the mean target value.


F0 = mean(y)

---

## Residual Calculation

The error of the current model is calculated:


Residual = Actual - Prediction

The next regression tree is trained on these residuals.

---

## Model Update

The new tree prediction is added using the learning rate:


New Prediction =
Old Prediction +
Learning Rate * Tree Prediction

---

# 8. Titanic Dataset Experiment

The model was evaluated on the Titanic survival prediction problem.

The dataset is a binary classification problem:


0 -> Not Survived
1 -> Survived

Features used after preprocessing and feature engineering included:

Numerical features:


Age
Fare
FamilySize
FarePerPerson
SibSp
Parch

Categorical features:


Sex
Embarked
FamilyCategory
Title
Deck

---

# 9. Initial Gradient Boosting Model

The first model was trained with:

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

Initial Results:
Metric	Score
Accuracy	0.8156
Precision	0.7903
Recall	0.7101
F1-score	0.7481


The model showed reasonable performance but required hyperparameter tuning.
10. Hyperparameter Tuning
Three important parameters were analyzed:
- Number of Trees
- Learning Rate
- Tree Depth
10.1 Effect of Number of Trees
Tested values:
10, 25, 50, 100, 150, 200

Results:
Number of Trees	Accuracy	F1-score
10	0.7933	0.6992
25	0.7877	0.7164
50	0.7933	0.7176
100	0.8156	0.7481
150	0.8156	0.7442
200	0.8156	0.7481


Increasing the number of trees improved performance initially.
After approximately 100 trees, the model reached a performance plateau.
Best value:
n_estimators = 100

10.2 Effect of Learning Rate
Tested values:
0.01, 0.05, 0.1, 0.2, 0.5

Results:
Learning Rate	Accuracy	F1-score
0.01	0.7933	0.6992
0.05	0.8156	0.7519
0.10	0.8156	0.7481
0.20	0.8101	0.7424
0.50	0.8101	0.7500


A very small learning rate caused underfitting because each tree contributed only a small correction.
The best F1-score was achieved with:
learning_rate = 0.05

10.3 Effect of Tree Depth
Tested values:
1,2,3,5,7

Results:
Max Depth	Accuracy	F1-score
1	0.7933	0.7132
2	0.8045	0.7328
3	0.8156	0.7519
5	0.8101	0.7463
7	0.8268	0.7737


Increasing depth improved performance on this dataset.
The best result was obtained with:
max_depth = 7

11. Final Gradient Boosting Model
The final model used:
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)


Final Performance:
Metric	Score
Accuracy	0.8268
Precision	0.7794
Recall	0.7681
F1-score	0.7737


12. Confusion Matrix Analysis
Final confusion matrix:
[[98, 12],
 [16, 53]]

Interpretation:
True Negative  = 98

False Positive = 12

False Negative = 16

True Positive  = 53

The model correctly identified most samples while maintaining a balanced trade-off between precision and recall.
13. Comparison with Other Classification Models
Final comparison:
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	0.827	0.806	0.725	0.763
KNN	0.821	0.794	0.725	0.758
Naive Bayes	0.771	0.656	0.855	0.742
Decision Tree	0.821	0.753	0.797	0.775
Random Forest	0.832	0.810	0.739	0.773
Gradient Boosting	0.827	0.779	0.768	0.774


14. Conclusion
Gradient Boosting successfully improved the baseline model by learning from previous prediction errors.
Important observations:
- Increasing the number of trees improved performance until reaching saturation.
- Learning rate controls the balance between learning speed and generalization.
- Tree depth strongly affects model complexity.
- Gradient Boosting achieved competitive performance compared with Random Forest and Decision Tree.
Although Random Forest achieved slightly higher accuracy on this dataset, Gradient Boosting provided a strong and balanced classification model with good precision and recall.
The experiment demonstrates how ensemble methods can improve predictive performance by combining multiple weak learners into a stronger model.