import numpy as np


class SVD:

    def __init__(
        self,
        n_components
    ):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None

    def fit(
        self,
        X
    ):

        eigenvalues, V = np.linalg.eigh(
            X.T @ X
        )

        idx = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = eigenvalues[idx]
        V = V[:, idx]

        S = np.sqrt(
            eigenvalues
        )

        V = V[:, :self.n_components]
        S = S[:self.n_components]

        U = (
            X @ V
        ) / S

        self.U = U
        self.S = S
        self.Vt = V.T

        return self

    def transform(
        self,
        X
    ):
        return X @ self.Vt.T

    def inverse_transform(
        self,
        X_transformed
    ):
        return (
            X_transformed @ self.Vt
        )

    def fit_transform(
        self,
        X
    ):
        self.fit(X)
        return self.transform(X)