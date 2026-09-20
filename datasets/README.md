# Datasets

# Datasets

Datasets are excluded from version control using `.gitignore` because
some datasets used in future experiments may be large.

This repository stores only:
- Dataset sources
- Download instructions
- Loading procedures
- Data preprocessing pipelines

Raw and processed data files are kept locally and are not committed to GitHub.


## Titanic Dataset
- Source: Kaggle Titanic Competition
- Link: https://www.kaggle.com/c/titanic
- Used in:
  - `05-machine-learning/00-data-preprocessing-and-eda/01_data_preprocessing.ipynb` -
  - `05-machine-learning/00-data-preprocessing-and-eda/02_exploratory_data_analysis.ipynb` -
  - `05-machine-learning/00-data-preprocessing-and-eda/03_feature_engineering.ipynb` -
  - `05-machine-learning/00-data-preprocessing-and-eda/04_feature_selection.ipynb` -
  - `05-machine-learning/01-supervised-learning/02-classification/01_logistic_regression.ipynb`
  - `05-machine-learning/01-supervised-learning/02-classification/02_k_nearest_neighbors.ipynb`
  - `05-machine-learning/01-supervised-learning/02-classification/03_naive_bayes.ipynb`
  - `05-machine-learning/01-supervised-learning/02-classification/04_decision_tree.ipynb`
  - `05-machine-learning/01-supervised-learning/02-classification/05_random_forest.ipynb`



## California Housing
- Source: scikit-learn built-in dataset
- Loading method:
  `fetch_california_housing()`
- Link: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- Used in:
  - `05-machine-learning/01-supervised-learning/regression/01_linear_regression.ipynb` - 

## Iris Dataset
- Source: UCI / Scikit-learn
- Link: https://archive.ics.uci.edu/ml/datasets/iris
- Used in:
  - `01-supervised-learning/Classification/02_k_nearest_neighbors.ipynb`

