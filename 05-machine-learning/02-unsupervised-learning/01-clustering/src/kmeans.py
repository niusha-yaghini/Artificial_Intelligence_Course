import numpy as np

class KMeans:

    def __init__(
        self,
        n_clusters=3,
        max_iterations=100,
        tolerance=1e-4
    ):

        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.centroids = None
        self.labels = None
        
    def _initialize_centroids(
        self,
        X
    ):

        indices = np.random.choice(
            X.shape[0],
            self.n_clusters,
            replace=False
        )

        return X[indices]
    
    def _euclidean_distance(
        self,
        X,
        centroid
    ):

        return np.sqrt(
            np.sum(
                (X-centroid)**2,
                axis=1
            )
        )
        
    def _assign_clusters(
        self,
        X
    ):
        distances = np.zeros(
            (
                X.shape[0],
                self.n_clusters
            )
        )

        for i, centroid in enumerate(
            self.centroids
        ):
            distances[:, i] = (
                self._euclidean_distance(
                    X,
                    centroid
                )
            )

        return np.argmin(
            distances,
            axis=1
        )
        
    def _update_centroids(
        self,
        X,
        labels
    ):
        centroids = np.zeros(
            (
                self.n_clusters,
                X.shape[1]
            )
        )

        for i in range(
            self.n_clusters
        ):
            points = X[
                labels == i
            ]
            centroids[i] = np.mean(
                points,
                axis=0
            )

        return centroids
    
    def fit(
        self,
        X
    ):
        self.centroids = (
            self._initialize_centroids(X)
        )

        for _ in range(
            self.max_iterations
        ):
            labels = (
                self._assign_clusters(X)
            )

            new_centroids = (
                self._update_centroids(
                    X,
                    labels
                )
            )

            difference = np.sum(
                abs(
                    new_centroids -
                    self.centroids
                )
            )

            self.centroids = new_centroids

            if difference < self.tolerance:
                break

        self.labels = (
            self._assign_clusters(X)
        )
        
    def predict(
        self,
        X
    ):
        return self._assign_clusters(
            X
        )
        
    def inertia(
        self,
        X
    ):
        total = 0

        for i in range(
            self.n_clusters
        ):
            points = X[
                self.labels == i
            ]

            total += np.sum(
                (
                    points -
                    self.centroids[i]
                ) ** 2
            )

        return total