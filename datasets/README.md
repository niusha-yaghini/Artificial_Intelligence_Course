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
  - `05-machine-learning/00-data-preprocessing-and-eda` -
  - `05-machine-learning/01-feature-engineering` -

## California Housing
- Source: scikit-learn built-in dataset
- Loading method:
  `fetch_california_housing()`
- Link: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- Used in:
  - `05-machine-learning/02-supervised-learning/regression` - 

## Iris Dataset
- Source: UCI / Scikit-learn
- Link: https://archive.ics.uci.edu/ml/datasets/iris
- Used in:
  - `01-supervised-learning/Classification/02_k_nearest_neighbors.ipynb`

