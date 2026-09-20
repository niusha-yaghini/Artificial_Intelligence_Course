import numpy as np


# ==========================
# Node Class
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
# Impurity Functions
# ==========================
def gini(y):
    classes, counts = np.unique(
        y,
        return_counts=True
    )
    impurity = 1
    for count in counts:
        probability = count / len(y)
        impurity -= probability ** 2

    return impurity

def entropy(y):
    classes, counts = np.unique(
        y,
        return_counts=True
    )
    entropy_value = 0

    for count in counts:
        probability = count / len(y)
        if probability > 0:
            entropy_value -= (
                probability *
                np.log2(probability)
            )

    return entropy_value

# ==========================
# Split Functions
# ==========================
def split_dataset(
    X,
    y,
    feature_index,
    threshold
):
    left_indices = (
        X[:, feature_index] <= threshold
    )
    right_indices = (
        X[:, feature_index] > threshold
    )

    X_left = X[left_indices]
    y_left = y[left_indices]

    X_right = X[right_indices]
    y_right = y[right_indices]

    return (
        X_left,
        y_left,
        X_right,
        y_right
    )

def weighted_impurity(
    y_left,
    y_right,
    impurity_function
):
    n = len(y_left) + len(y_right)

    weighted = (
        (len(y_left) / n)
        *
        impurity_function(y_left)
        +
        (len(y_right) / n)
        *
        impurity_function(y_right)
    )

    return weighted

def best_split(
    X,
    y,
    impurity_function
):
    best_feature = None
    best_threshold = None
    best_impurity = float("inf")

    n_features = X.shape[1]

    for feature_index in range(n_features):
        thresholds = np.unique(
            X[:, feature_index]
        )
        for threshold in thresholds:
            (
                X_left,
                y_left,
                X_right,
                y_right
            ) = split_dataset(
                X,
                y,
                feature_index,
                threshold
            )

            # Avoid empty split
            if (
                len(y_left) == 0
                or
                len(y_right) == 0
            ):
                continue

            impurity = weighted_impurity(
                y_left,
                y_right,
                impurity_function
            )

            if impurity < best_impurity:
                best_impurity = impurity
                best_feature = feature_index
                best_threshold = threshold

    return (
        best_feature,
        best_threshold,
        best_impurity
    )

# ==========================
# Decision Tree Class
# ==========================
class DecisionTree:
    def __init__(
        self,
        criterion="gini",
        max_depth=5,
        min_samples_split=2
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # ----------------------
    # Select impurity
    # ----------------------
    def _get_impurity_function(self):
        if self.criterion == "gini":
            return gini
        elif self.criterion == "entropy":
            return entropy
        else:
            raise ValueError(
                "criterion must be 'gini' or 'entropy'"
            )

    # ----------------------
    # Training
    # ----------------------
    def fit(
        self,
        X,
        y
    ):
        impurity_function = (
            self._get_impurity_function()
        )
        self.root = self._build_tree(
            X,
            y,
            depth=0,
            impurity_function=impurity_function
        )

    # ----------------------
    # Recursive Tree Builder
    # ----------------------
    def _build_tree(
        self,
        X,
        y,
        depth,
        impurity_function
    ):
        num_samples = len(y)
        num_classes = len(
            np.unique(y)
        )

        # stopping criteria
        if (
            depth >= self.max_depth
            or
            num_samples < self.min_samples_split
            or
            num_classes == 1
        ):
            leaf_value = np.argmax(
                np.bincount(y)
            )
            return Node(
                value=leaf_value
            )

        (
            best_feature,
            best_threshold,
            best_impurity

        ) = best_split(
            X,
            y,
            impurity_function
        )

        # no useful split
        if best_feature is None:
            leaf_value = np.argmax(
                np.bincount(y)
            )
            return Node(
                value=leaf_value
            )

        (
            X_left,
            y_left,
            X_right,
            y_right
        ) = split_dataset(
            X,
            y,
            best_feature,
            best_threshold
        )

        # Recursive construction
        left_child = self._build_tree(
            X_left,
            y_left,
            depth + 1,
            impurity_function
        )

        right_child = self._build_tree(
            X_right,
            y_right,
            depth + 1,
            impurity_function
        )

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
        
    def _predict_sample(
        self,
        x,
        node
    ):
        # Leaf node
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(
                x,
                node.left
            )
        else:
            return self._predict_sample(
                x,
                node.right
            )
            
    def predict(
        self,
        X
    ):
        predictions = []

        for x in X:
            prediction = self._predict_sample(
                x,
                self.root
            )
            predictions.append(
                prediction
            )

        return np.array(
            predictions
        )
        
    def print_tree(
        self,
        node=None,
        depth=0
    ):
        if node is None:
            node = self.root

        # Leaf node
        if node.value is not None:
            print(
                "  " * depth
                +
                f"Leaf: {node.value}"
            )
            return

        # Decision node
        print(
            "  " * depth
            +
            f"Feature {node.feature} <= {node.threshold}"
        )
        print(
            "  " * depth
            +
            "Left:"
        )
        self.print_tree(
            node.left,
            depth + 1
        )
        print(
            "  " * depth
            +
            "Right:"
        )
        self.print_tree(
            node.right,
            depth + 1
        )