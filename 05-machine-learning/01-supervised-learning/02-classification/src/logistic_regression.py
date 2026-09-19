import numpy as np


class LogisticRegressionGD:

    def __init__(
        self,
        learning_rate=0.01,
        epochs=1000
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None

        self.loss_history = []


    def sigmoid(
        self,
        z
    ):

        return 1 / (
            1 + np.exp(-z)
        )


    def fit(
        self,
        X,
        y
    ):

        n_samples, n_features = X.shape


        # Initialize parameters

        self.weights = np.zeros(
            n_features
        )

        self.bias = 0


        # Gradient Descent

        for epoch in range(
            self.epochs
        ):


            # Linear model

            z = (
                X @ self.weights
                + self.bias
            )


            # Probability

            y_pred = self.sigmoid(
                z
            )


            # Error

            error = (
                y_pred - y
            )


            # Gradients

            dw = (
                (1 / n_samples)
                *
                (X.T @ error)
            )


            db = (
                (1 / n_samples)
                *
                np.sum(error)
            )


            # Update parameters

            self.weights -= (
                self.learning_rate
                *
                dw
            )


            self.bias -= (
                self.learning_rate
                *
                db
            )


            # Loss

            loss = -np.mean(
                y * np.log(y_pred + 1e-15)
                +
                (1-y)
                *
                np.log(
                    1-y_pred + 1e-15
                )
            )


            self.loss_history.append(
                loss
            )


    def predict_proba(
        self,
        X
    ):

        z = (
            X @ self.weights
            + self.bias
        )


        return self.sigmoid(
            z
        )


    def predict(
        self,
        X,
        threshold=0.5
    ):

        probabilities = self.predict_proba(
            X
        )


        return (
            probabilities >= threshold
        ).astype(int)