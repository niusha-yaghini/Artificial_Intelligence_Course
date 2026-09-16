Data Preprocessing and Exploratory Data Analysis

This section focuses on understanding raw data, preparing it for machine learning models, and extracting meaningful insights through exploratory analysis.

The Titanic dataset was used as a case study to demonstrate a complete data preparation workflow.

01 — Data Preprocessing
Objective

The goal of this notebook was to transform raw data into a clean and model-ready format while preventing common machine learning issues such as data leakage.

Workflow

The preprocessing pipeline included:

1. Dataset Inspection

Initial analysis was performed to understand:

Dataset shape
Feature types
Numerical and categorical variables
Missing values
Basic statistical properties
2. Feature Selection

The initial feature set was selected as:

Numerical features:

Pclass
Age
SibSp
Parch
Fare

Categorical features:

Sex
Embarked

The following variables were removed during initial preprocessing:

Feature	Reason
PassengerId	Identifier without predictive meaning
Name	Requires feature extraction
Ticket	High-cardinality categorical variable
Cabin	High percentage of missing values

These features were reserved for possible feature engineering steps.

3. Train/Test Split

The dataset was divided into training and testing subsets:

80% Training data
20% Testing data

Stratified splitting was applied to preserve the survival class distribution.

4. Missing Value Handling

Missing values were handled using statistics learned only from the training data.

Numerical:

Age → Median imputation

Categorical:

Embarked → Most frequent value imputation

This prevents information leakage from the test set.

5. Categorical Encoding

Categorical variables were transformed using:

One-Hot Encoding

Applied to:

Sex
Embarked

with:

handle_unknown="ignore"

to handle unseen categories during inference.

6. Feature Scaling

Numerical variables were standardized using:

$$ z=\frac{x-\mu}{\sigma} $$

using StandardScaler.

Scaling parameters were learned only from training data.

7. Pipeline Implementation

The preprocessing workflow was implemented in two ways:

Manual Implementation

Individual steps were performed explicitly:

Missing value handling
Encoding
Feature combination
Scaling

This approach was used to understand the internal process.

Scikit-learn Pipeline

A production-style pipeline was created using:

SimpleImputer
StandardScaler
OneHotEncoder
ColumnTransformer
Pipeline

Advantages:

Reproducible preprocessing
Reduced code duplication
Prevention of data leakage
Easy integration with ML models
02 — Exploratory Data Analysis
Objective

EDA was performed to understand the underlying patterns, relationships, and statistical properties of the dataset before modeling.

Analysis Performed
Dataset Overview

Examined:

Dataset dimensions
Data types
Statistical summaries
Missing Value Analysis

Investigated missing data patterns.

Main observations:

Age contains moderate missing values.
Cabin contains a large percentage of missing values.
Embarked has only a small number of missing samples.
Numerical Feature Analysis

Analyzed:

Age
Fare
SibSp
Parch

using:

Distribution plots
Outlier analysis
Statistical summaries

Main findings:

Fare has a highly right-skewed distribution.
Most passengers traveled alone or with a small number of family members.
Age distribution is concentrated around young and middle-aged passengers.
Categorical Feature Analysis

Analyzed:

Sex
Pclass
Embarked

Main observations:

The dataset contains more male passengers.
Third class represents the largest passenger group.
Southampton is the dominant embarkation point.
Feature–Target Relationship Analysis

The relationship between features and survival outcome was investigated.

Key Findings
Sex

Female passengers showed substantially higher survival rates compared with male passengers.

Passenger Class

Higher passenger classes were associated with higher survival rates.

Fare

Higher fares were associated with increased survival probability.

This relationship may partially reflect the effect of passenger class.

Family Features

SibSp and Parch showed differences between survival groups, suggesting that family-related information may provide predictive value.

Statistical Analysis

Statistical tests were performed to support EDA observations.

Chi-Square Test

Used for categorical features:

Sex
Pclass
Embarked

Results:

All investigated categorical features showed statistically significant associations with survival.

Mann-Whitney U Test

Used for numerical features:

Age
Fare
SibSp
Parch

Results:

Feature	Result
Age	No significant difference
Fare	Significant difference
SibSp	Significant difference
Parch	Significant difference
Final Outcome

After completing this section:

Raw data understanding was achieved.
A leakage-safe preprocessing workflow was created.
Model-ready data transformation was implemented.
Important relationships between features and survival were identified.
Potential feature engineering opportunities were discovered.

The next step is Feature Engineering, where new informative features will be created from existing variables to improve machine learning performance.