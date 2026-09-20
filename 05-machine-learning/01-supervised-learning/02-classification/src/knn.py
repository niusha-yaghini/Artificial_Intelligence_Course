import numpy as np


class KNearestNeighbors:

    def __init__(
        self,
        k=5
    ):
        self.k = k

        self.X_train = None
        self.y_train = None


    def fit(
        self,
        X,
        y
    ):
        self.X_train = np.asarray(
            X,
            dtype=float
        )

        self.y_train = np.asarray(
            y
        )


    def _euclidean_distance(
        self,
        x1,
        x2
    ):
        return np.sqrt(
            np.sum(
                (x1 - x2) ** 2
            )
        )


    def predict(
        self,
        X
    ):
        X = np.asarray(
            X,
            dtype=float
        )

        predictions = []

        for x in X:

            distances = []

            for x_train in self.X_train:

                distance = self._euclidean_distance(
                    x,
                    x_train
                )

                distances.append(
                    distance
                )


            nearest_indices = np.argsort(
                distances
            )[:self.k]


            nearest_labels = self.y_train[
                nearest_indices
            ]


            values, counts = np.unique(
                nearest_labels,
                return_counts=True
            )


            predicted_class = values[
                np.argmax(counts)
            ]


            predictions.append(
                predicted_class
            )


        return np.array(
            predictions
        )
        
        
        
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)



# import pandas as pd
# from pathlib import Path

# csv_path = Path(__file__).parent / "titanic_feature_engineered.csv"

# df = pd.read_csv(csv_path)

# X = df.drop(
#     "Survived",
#     axis=1
# )

# y = df["Survived"]

# numeric_features = X.select_dtypes(
#     include=[
#         "int64",
#         "float64"
#     ]
# ).columns.tolist()


# categorical_features = X.select_dtypes(
#     include=[
#         "object"
#     ]
# ).columns.tolist()
        
        
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )
        
# numeric_transformer = Pipeline(
#     steps=[
        
#         (
#             "imputer",
#             SimpleImputer(
#                 strategy="median"
#             )
#         ),

#         (
#             "scaler",
#             StandardScaler()
#         )
#     ]
# )

# categorical_transformer = Pipeline(
#     steps=[

#         (
#             "imputer",
#             SimpleImputer(
#                 strategy="most_frequent"
#             )
#         ),

#         (
#             "encoder",
#             OneHotEncoder(
#                 handle_unknown="ignore"
#             )
#         )
#     ]
# )

# preprocessor = ColumnTransformer(
#     transformers=[

#         (
#             "num",
#             numeric_transformer,
#             numeric_features
#         ),
#         (
#             "cat",
#             categorical_transformer,
#             categorical_features
#         )
#     ]
# )

# X_train_processed = preprocessor.fit_transform(
#     X_train
# )

# X_test_processed = preprocessor.transform(
#     X_test
# )
        
        
        
# knn_model = KNearestNeighbors(
#     k=5
# )

# knn_model.fit(
#     X_train_processed,
#     y_train.values
# )

