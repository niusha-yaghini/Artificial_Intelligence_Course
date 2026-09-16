# Datasets

Datasets are intentionally excluded from version control using `.gitignore`
because some datasets used in future experiments may be large.

Each notebook provides:
- Dataset source
- Download instructions
- Loading procedure

Small and reproducible datasets may be loaded automatically from libraries
such as scikit-learn.


## Titanic Dataset
- Source: Kaggle Titanic Competition
- Link: https://www.kaggle.com/c/titanic
- Local filename: `raw/titanic.csv`
- Used in:
  - `00-data-preprocessing-and-eda/01_exploratory-data-analysis/`

## California Housing
- Source: Scikit-learn
- Link: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- Used in:
  - `01-supervised-learning/regression` - 

## Iris Dataset
- Source: UCI / Scikit-learn
- Link: https://archive.ics.uci.edu/ml/datasets/iris
- Local filename: `raw/iris.csv`
- Used in:
  - `01-supervised-learning/Classification/02_k_nearest_neighbors.ipynb`

