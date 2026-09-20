import numpy as np
from collections import Counter


# ==========================
# Node
# ==========================
class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# ==========================
# Decision Tree for RF
# ==========================
class RandomForestTree:
    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        max_features=None
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    # Fit
    def fit(
        self,
        X,
        y
    ):

        self.root = self._grow_tree(
            X,
            y,
            depth=0
        )

    # Prediction
    def predict(
        self,
        X
    ):
        predictions = []

        for sample in X:
            predictions.append(
                self._traverse_tree(
                    sample,
                    self.root
                )
            )

        return np.array(
            predictions
        )

    def _traverse_tree(
        self,
        x,
        node
    ):
        if node.value is not None:
            return node.value

        if (
            x[node.feature]
            <=
            node.threshold
        ):
            return self._traverse_tree(
                x,
                node.left
            )
        else:
            return self._traverse_tree(
                x,
                node.right
            )

    # Build Tree
    def _grow_tree(
        self,
        X,
        y,
        depth
    ):
        n_samples, n_features = X.shape

        if n_samples == 0:
            return Node(
                value=0
            )

        n_classes = len(
            np.unique(y)
        )

        # stopping criteria
        if (
            n_classes == 1
            or
            n_samples < self.min_samples_split
            or
            (
                self.max_depth is not None

                and

                depth >= self.max_depth
            )
        ):
            return Node(
                value=self._majority_class(y)
            )

        (
            best_feature,
            best_threshold
        ) = self._best_split(
            X,
            y
        )


        if best_feature is None:
            return Node(
                value=self._majority_class(y)
            )

        left_mask = (
            X[:, best_feature]
            <=
            best_threshold
        )

        right_mask = (
            X[:, best_feature]
            >
            best_threshold
        )

        left_child = self._grow_tree(
            X[left_mask],
            y[left_mask],
            depth+1
        )

        right_child = self._grow_tree(
            X[right_mask],
            y[right_mask],
            depth+1
        )

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    # Best Split
    def _best_split(
        self,
        X,
        y
    ):
        n_features = X.shape[1]

        # Feature Randomness
        if self.max_features is None:
            features = np.arange(
                n_features
            )
        else:
            features = np.random.choice(
                n_features,
                self.max_features,
                replace=False
            )

        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in features:
            values = np.unique(
                X[:,feature]
            )

            thresholds = (
                values[:-1]
                +
                values[1:]
            ) / 2

            for threshold in thresholds:
               
                left = X[:, feature] <= threshold
                right = X[:, feature] > threshold

                if (
                    np.sum(left) == 0
                    or
                    np.sum(right) == 0
                ):
                    continue

                gain = self._information_gain(
                    y,
                    X[:,feature],
                    threshold
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return (
            best_feature,
            best_threshold
        )

    # Information Gain
    def _information_gain(
        self,
        y,
        feature_values,
        threshold
    ):
        parent_impurity = (
            self._impurity(y)
        )

        left = (
            feature_values <= threshold
        )

        right = (
            feature_values > threshold
        )

        if (
            np.sum(left)==0
            or
            np.sum(right)==0
        ):
            return 0

        n=len(y)
        n_left=np.sum(left)
        n_right=np.sum(right)

        child_impurity = (
            (n_left/n)
            *
            self._impurity(
                y[left]
            )
            +
            (n_right/n)
            *
            self._impurity(
                y[right]
            )
        )

        return (
            parent_impurity
            -
            child_impurity
        )

    # Impurity Selector
    def _impurity(
        self,
        y
    ):
        if self.criterion=="gini":
            return self._gini(y)
        elif self.criterion=="entropy":
            return self._entropy(y)
        else:
            raise ValueError(
                "criterion must be gini or entropy"
            )

    # Gini
    def _gini(
        self,
        y
    ):
        classes, counts = np.unique(
            y,
            return_counts=True
        )

        impurity=1

        for count in counts:
            p=count/len(y)
            impurity -= p**2

        return impurity

    # Entropy
    def _entropy(
        self,
        y
    ):
        classes, counts = np.unique(
            y,
            return_counts=True
        )

        entropy=0

        for count in counts:
            p=count/len(y)

            if p>0:
                entropy -= (
                    p*np.log2(p)
                )

        return entropy

    # Majority Class
    def _majority_class(
        self,
        y
    ):
        return Counter(y).most_common(1)[0][0]


# ==========================
# Random Forest
# ==========================
class RandomForest:
    def __init__(
        self,
        n_estimators=10,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        max_features=None
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    # Training
    def fit(
        self,
        X,
        y
    ):
        self.trees=[]

        for i in range(
            self.n_estimators
        ):

            # Bootstrap Sampling
            indices = np.random.choice(
                len(X),
                len(X),
                replace=True
            )

            X_sample = X[
                indices
            ]

            y_sample = y[
                indices
            ]

            tree = RandomForestTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )

            tree.fit(
                X_sample,
                y_sample
            )

            self.trees.append(
                tree
            )

    # Prediction
    def predict(
        self,
        X
    ):
        predictions=[]

        for tree in self.trees:
            predictions.append(
                tree.predict(X)
            )

        predictions=np.array(
            predictions
        )

        final_predictions=[]

        for sample_votes in predictions.T:
            vote = Counter(
                sample_votes
            ).most_common(1)[0][0]
            final_predictions.append(
                vote
            )

        return np.array(
            final_predictions
        )