import numpy as np


class PCA:

    def __init__(
        self,
        n_components
    ):

        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None


    def fit(
        self,
        X
    ):

        # 1. Centering
        self.mean = np.mean(
            X,
            axis=0
        )

        X_centered = X - self.mean


        # 2. Covariance Matrix
        covariance = np.cov(
            X_centered,
            rowvar=False
        )


        # 3. Eigen Decomposition
        eigenvalues, eigenvectors = np.linalg.eig(
            covariance
        )


        # 4. Sort by eigenvalues
        indices = np.argsort(
            eigenvalues
        )[::-1]


        eigenvalues = eigenvalues[
            indices
        ]

        eigenvectors = eigenvectors[
            :,
            indices
        ]


        # 5. Select top components
        self.components = eigenvectors[
            :,
            :self.n_components
        ].T


        # 6. Explained variance
        self.explained_variance = eigenvalues[
            :self.n_components
        ]


        self.explained_variance_ratio = (
            self.explained_variance /
            np.sum(eigenvalues)
        )


        return self


    def transform(
        self,
        X
    ):

        X_centered = X - self.mean


        return X_centered @ self.components.T


    def fit_transform(
        self,
        X
    ):

        self.fit(
            X
        )

        return self.transform(
            X
        )
        
    def inverse_transform(
        self,
        X_transformed
    ):

        return (
            X_transformed @ self.components
            +
            self.mean
        )