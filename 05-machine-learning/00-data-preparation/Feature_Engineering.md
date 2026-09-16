03 - Feature Engineering and Feature Selection

This section focuses on transforming raw data into meaningful representations and identifying the most informative features for machine learning models.

Feature engineering and feature selection are essential steps in the machine learning pipeline because the quality of input features directly affects model performance, generalization ability, and interpretability.

The Titanic dataset is used as a practical case study to demonstrate different feature transformation and selection techniques.

01 - Feature Engineering
Objective

The goal of feature engineering is to create new informative features from existing raw variables by incorporating domain knowledge and extracting hidden patterns from the data.

Instead of directly using raw variables, meaningful representations are generated to better capture relationships between passenger characteristics and survival outcomes.

Techniques Applied
1. Family-Based Features

Original features:

SibSp
Parch

were combined to create:

FamilySize

Represents the total number of family members traveling together:

$$ FamilySize = SibSp + Parch + 1 $$

This feature provides more meaningful information about passenger travel groups.

IsAlone

A binary feature indicating whether a passenger traveled alone:

$$ IsAlone = \begin{cases} 1 & FamilySize = 1\\ 0 & FamilySize > 1 \end{cases} $$

This captures the effect of traveling without family members.

FamilyCategory

Passengers were grouped into different family-size categories:

Alone
Small Family
Large Family

This transformation helps capture possible nonlinear relationships between family size and survival probability.

2. Name-Based Feature Extraction

The original Name variable contains hidden demographic information.

Passenger titles were extracted:

Examples:

Mr
Mrs
Miss
Master
Rare titles

The extracted feature:

Title

captures information related to:

Gender
Age group
Social status
3. Cabin-Based Feature Extraction

Due to the high percentage of missing values in the original Cabin feature, the complete cabin number was not directly used.

Instead, the first character was extracted:

Cabin → Deck

Example:

C85 → C
E46 → E

Missing cabin values were assigned:

Unknown

This preserves useful information about passenger location on the ship.

4. Fare Transformation

A new feature was created:

FarePerPerson
$$ FarePerPerson=\frac{Fare}{FamilySize} $$

This represents the approximate fare paid per individual rather than the total ticket price.

This transformation reduces the effect of group tickets and provides a more representative measure of passenger economic status.

Output

After feature engineering, the dataset was transformed from raw passenger information into a feature-engineered dataset:

titanic_feature_engineered.csv

The final dataset contains:

Original informative features
Newly created features
Target variable (Survived)
<br>
02 - Feature Selection
Objective

Feature selection aims to identify the most informative features and remove irrelevant or redundant variables before model training.

The goal is to improve:

Model generalization
Training efficiency
Interpretability
Reduction of unnecessary complexity
Feature Selection Methods

Several statistical and information-based methods were applied.

1. Correlation Analysis

Pearson correlation was used to analyze linear relationships between numerical features and the target variable.

Analyzed features:

Age
Fare
FamilySize
FarePerPerson
SibSp
Parch

The analysis showed that:

Fare and FarePerPerson had stronger positive relationships with survival.
Some variables with weak correlation still contained useful nonlinear information.

Therefore, correlation alone was not used as the only selection criterion.

2. Chi-Square Test

Chi-square testing was applied to categorical features to evaluate dependency between categorical variables and survival.

Analyzed features:

Sex
Embarked
FamilyCategory
Title
Deck

All tested categorical features showed statistically significant relationships with survival.

The strongest associations were observed for:

Title
Sex
3. Mutual Information

Mutual Information was used to measure how much information each feature provides about the survival target.

Unlike correlation, Mutual Information can capture nonlinear relationships.

Results showed that the most informative features were:

Numerical Features
FarePerPerson
Fare
FamilySize
Categorical Features
Title
Sex
Deck
Final Feature Selection Strategy

Feature selection decisions were not based on a single metric.

The final feature set was determined by considering:

Statistical significance
Information content
Domain knowledge
Redundancy between features

The selected features will be used in later machine learning models.

Summary

This section demonstrates a complete feature preparation workflow:

Raw Dataset

      ↓

Feature Engineering

      ↓

Feature Selection

      ↓

Machine Learning Models

The developed pipeline creates meaningful features from raw data and identifies the most valuable information before applying supervised learning algorithms.

------------------------------------

In Future:

03-feature-engineering

├── Feature Transformation
│   ├── Log transform
│   ├── Box-Cox
│   └── Power transform
│
├── Feature Construction
│   ├── Polynomial Features
│   ├── Interaction Features
│   └── Domain Features
│
├── Feature Selection
│   ├── Filter Methods
│   ├── Wrapper Methods
│   └── Embedded Methods
│
└── Feature Extraction
    ├── PCA
    └── Autoencoder