import numpy as np

class DBSCAN:
    def __init__(
        self,
        eps=0.5,
        min_samples=5
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        
    def _distance(
        self,
        a,
        b
    ):
        return np.sqrt(
            np.sum(
                (a-b)**2
            )
        )
        
    def _region_query(
        self,
        X,
        index
    ):
        neighbors = []
        for i, point in enumerate(X):
            distance = self._distance(
                X[index],
                point
            )
            if distance <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def _expand_cluster(
        self,
        X,
        labels,
        point_index,
        neighbors,
        cluster_id
    ):
        labels[point_index] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_index = neighbors[i]
            if labels[neighbor_index] == -99:
                labels[neighbor_index] = cluster_id
                neighbor_neighbors = (
                    self._region_query(
                        X,
                        neighbor_index
                    )
                )
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors += neighbor_neighbors
            i += 1

    def fit(
        self,
        X
    ):

        n_samples = X.shape[0]

        labels = np.full(
            n_samples,
            -99
        )

        cluster_id = 0


        for i in range(n_samples):

            if labels[i] != -99:
                continue


            neighbors = self._region_query(
                X,
                i
            )


            if len(neighbors) < self.min_samples:

                labels[i] = -1


            else:

                self._expand_cluster(
                    X,
                    labels,
                    i,
                    neighbors,
                    cluster_id
                )

                cluster_id += 1


        self.labels = labels