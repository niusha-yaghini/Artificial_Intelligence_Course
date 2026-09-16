import numpy as np


class LinearRegressionGD:

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


    def fit(
        self,
        X,
        y
    ):

        n_samples, n_features = X.shape


        self.weights = np.zeros(
            n_features
        )

        self.bias = 0


        for epoch in range(
            self.epochs
        ):

            y_pred = (
                np.dot(
                    X,
                    self.weights
                )
                +
                self.bias
            )


            error = y - y_pred


            loss = np.mean(
                error ** 2
            )


            self.loss_history.append(
                loss
            )


            dw = (
                -2 *
                np.dot(
                    X.T,
                    error
                )
                /
                n_samples
            )


            db = (
                -2 *
                np.mean(error)
            )


            self.weights -= (
                self.learning_rate *
                dw
            )


            self.bias -= (
                self.learning_rate *
                db
            )
            
    def predict(
        self,
        X
    ):

        y_pred = (
            np.dot(
                X,
                self.weights
            )
            +
            self.bias
        )

        return y_pred