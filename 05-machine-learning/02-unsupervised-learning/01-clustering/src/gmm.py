import numpy as np


class GaussianMixture:

    def __init__(
        self,
        n_components=2,
        max_iterations=100,
        tolerance=1e-4
    ):

        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.means = None
        self.covariances = None
        self.weights = None

        self.labels = None
        self.responsibilities = None
        
    def _gaussian_probability(
        self,
        X,
        mean,
        covariance
    ):
        n_features = X.shape[1]
        covariance_det = np.linalg.det(
            covariance
        )
        covariance_inv = np.linalg.inv(
            covariance
        )
        diff = X - mean
        exponent = np.exp(
            -0.5 *
            np.sum(
                (diff @ covariance_inv) * diff,
                axis=1
            )
        )
        denominator = np.sqrt(
            (2*np.pi)**n_features *
            covariance_det
        )
        return exponent / denominator
    
    def _initialize_parameters(
        self,
        X
    ):
        n_samples, n_features = X.shape
        indices = np.random.choice(
            n_samples,
            self.n_components,
            replace=False
        )
        self.means = X[indices]
        self.covariances = np.array(
            [
                np.eye(n_features)
                for _ in range(self.n_components)
            ]
        )
        self.weights = np.ones(
            self.n_components
        ) / self.n_components
        
    def _expectation(
        self,
        X
    ):
        n_samples = X.shape[0]
        responsibilities = np.zeros(
            (
                n_samples,
                self.n_components
            )
        )
        for k in range(
            self.n_components
        ):
            probabilities = self._gaussian_probability(
                X,
                self.means[k],
                self.covariances[k]
            )
            responsibilities[:, k] = (
                self.weights[k] *
                probabilities
            )
        responsibilities /= (
            responsibilities.sum(
                axis=1,
                keepdims=True
            )
        )
        return responsibilities
    
    def _maximization(
        self,
        X,
        responsibilities
    ):

        n_samples, n_features = X.shape

        Nk = responsibilities.sum(
            axis=0
        )

        self.weights = (
            Nk / n_samples
        )


        self.means = np.zeros(
            (
                self.n_components,
                n_features
            )
        )


        for k in range(
            self.n_components
        ):

            self.means[k] = (
                np.sum(
                    responsibilities[:,k,None] * X,
                    axis=0
                )
                /
                Nk[k]
            )


        self.covariances = []


        for k in range(
            self.n_components
        ):

            diff = X - self.means[k]

            weighted_diff = (
                responsibilities[:,k,None]
                *
                diff
            )

            covariance = (
                weighted_diff.T @ diff
                /
                Nk[k]
            )

            covariance += (
                np.eye(n_features)
                * 1e-6
            )

            self.covariances.append(
                covariance
            )


        self.covariances = np.array(
            self.covariances
        )
    
    def fit(
        self,
        X
    ):
        self._initialize_parameters(
            X
        )
        for _ in range(
            self.max_iterations
        ):
            old_means = self.means.copy()
            responsibilities = self._expectation(
                X
            )
            self._maximization(
                X,
                responsibilities
            )
            change = np.linalg.norm(
                self.means - old_means
            )
            if change < self.tolerance:
                break
        self.responsibilities = responsibilities
        self.labels = np.argmax(
            responsibilities,
            axis=1
        )