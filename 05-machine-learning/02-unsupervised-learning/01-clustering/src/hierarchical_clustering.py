import numpy as np

class HierarchicalClustering:

    def __init__(
        self,
        n_clusters=1
    ):

        self.n_clusters = n_clusters

        self.labels = None
        self.clusters = None
        self.history = []
        
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
        
    def _cluster_distance(
        self,
        X,
        cluster1,
        cluster2
    ):

        distances = []

        for i in cluster1:
            for j in cluster2:
                distances.append(
                    self._distance(
                        X[i],
                        X[j]
                    )
                )

        return min(distances)
    
    def _find_closest_clusters(
        self,
        X,
        clusters
    ):

        min_distance = float("inf")
        pair = None

        for i in range(len(clusters)):
            for j in range(
                i+1,
                len(clusters)
            ):
                distance = self._cluster_distance(
                    X,
                    clusters[i],
                    clusters[j]
                )

                if distance < min_distance:
                    min_distance = distance
                    pair = (i,j)

        return pair
    
    def _merge_clusters(
        self,
        clusters,
        cluster1,
        cluster2
    ):
        new_cluster = (
            cluster1 +
            cluster2
        )

        clusters.remove(cluster1)
        clusters.remove(cluster2)

        clusters.append(
            new_cluster
        )
        
    def fit(
        self,
        X
    ):
        n_samples = X.shape[0]

        clusters = [
            [i]
            for i in range(n_samples)
        ]

        self.history = []

        while len(clusters) > self.n_clusters:
            pair = self._find_closest_clusters(
                X,
                clusters
            )

            cluster1 = clusters[pair[0]]
            cluster2 = clusters[pair[1]]

            distance = self._cluster_distance(
                X,
                cluster1,
                cluster2
            )

            self.history.append(
                [
                    cluster1.copy(),
                    cluster2.copy(),
                    distance
                ]
            )

            self._merge_clusters(
                clusters,
                cluster1,
                cluster2
            )

        self.clusters = clusters

        self.labels = np.zeros(
            n_samples,
            dtype=int
        )

        for label, cluster in enumerate(clusters):
            for index in cluster:
                self.labels[index] = label 
                
    def get_linkage_matrix(
        self
    ):
        linkage_matrix = []
        cluster_ids = {}
        next_id = 0

        for merge in self.history:
            cluster1, cluster2, distance = merge

            key1 = tuple(cluster1)
            key2 = tuple(cluster2)

            if key1 not in cluster_ids:
                cluster_ids[key1] = next_id
                next_id += 1

            if key2 not in cluster_ids:
                cluster_ids[key2] = next_id
                next_id += 1

            id1 = cluster_ids[key1]
            id2 = cluster_ids[key2]

            new_cluster = (
                cluster1 +
                cluster2
            )

            cluster_ids[
                tuple(new_cluster)
            ] = next_id

            linkage_matrix.append(
                [
                    id1,
                    id2,
                    distance,
                    len(new_cluster)
                ]
            )

            next_id += 1

        return np.array(
            linkage_matrix
        )
                