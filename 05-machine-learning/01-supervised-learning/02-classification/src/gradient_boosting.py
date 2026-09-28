import numpy as np


class RegressionNode:

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


class RegressionTree:

    def __init__(
        self,
        max_depth=3,
        min_samples_split=2
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(
            X,
            y,
            0
        )

    def predict(self, X):
        return np.array(
            [
                self._predict_sample(x, self.root)
                for x in X
            ]
        )

    def _predict_sample(self, x, node):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(
                x,
                node.left
            )

        return self._predict_sample(
            x,
            node.right
        )

    def _grow_tree(self, X, y, depth):

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
        ):
            return RegressionNode(
                value=np.mean(y)
            )


        feature, threshold = self._best_split(
            X,
            y
        )

        if feature is None:
            return RegressionNode(
                value=np.mean(y)
            )

        left = X[:, feature] <= threshold
        right = X[:, feature] > threshold

        return RegressionNode(
            feature=feature,
            threshold=threshold,
            left=self._grow_tree(
                X[left],
                y[left],
                depth + 1
            ),
            right=self._grow_tree(
                X[right],
                y[right],
                depth + 1
            )
        )

    def _best_split(self, X, y):

        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):

            values = np.unique(
                X[:, feature]
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
                    not left.any()
                    or
                    not right.any()
                ):
                    continue

                mse = self._mse(
                    y[left],
                    y[right]
                )

                if mse < best_mse:

                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return (
            best_feature,
            best_threshold
        )

    def _mse(self, left_y, right_y):

        left_error = np.mean(
            (left_y - np.mean(left_y)) ** 2
        )

        right_error = np.mean(
            (right_y - np.mean(right_y)) ** 2
        )

        return (
            len(left_y) * left_error
            +
            len(right_y) * right_error
        ) / (
            len(left_y)
            +
            len(right_y)
        )
        
    def print_tree(self, node=None, depth=0):

        if node is None:
            node = self.root

        if node.value is not None:
            print(
                "  " * depth,
                "Leaf:",
                node.value
            )
            return

        print(
            "  " * depth,
            f"Feature {node.feature} <= {node.threshold}"
        )

        self.print_tree(
            node.left,
            depth + 1
        )

        self.print_tree(
            node.right,
            depth + 1
        )
        
        
class GradientBoosting:

    def __init__(
        self,
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3
    ):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):

        self.initial_prediction = np.mean(y)

        prediction = np.full(
            len(y),
            self.initial_prediction
        )

        for i in range(self.n_estimators):
            residual = y - prediction
            
            print(
                "Tree:",
                i+1,
                "Residual MSE:",
                np.mean(residual**2)
            )

            tree = RegressionTree(
                max_depth=self.max_depth
            )

            tree.fit(
                X,
                residual
            )

            update = tree.predict(
                X
            )

            prediction += (
                self.learning_rate
                *
                update
            )

            self.trees.append(
                tree
            )

    def predict(self, X):
        prediction = np.full(
            X.shape[0],
            self.initial_prediction
        )

        for tree in self.trees:
            prediction += (
                self.learning_rate
                *
                tree.predict(X)
            )

        return prediction