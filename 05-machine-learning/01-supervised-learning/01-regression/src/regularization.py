import numpy as np


class RidgeRegressionGD:

    def __init__(
        self,
        learning_rate=0.001,
        epochs=1000,
        lambda_=0.1
    ):
        """
        Ridge Regression using Gradient Descent

        Parameters
        ----------
        learning_rate : float
            Step size for Gradient Descent

        epochs : int
            Number of training iterations

        lambda_ : float
            Regularization strength
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_

        self.weights = None
        self.bias = None

        self.loss_history = []


    def fit(
        self,
        X,
        y
    ):
        """
        Train Ridge Regression model

        Parameters
        ----------
        X : numpy array
            Input features

        y : numpy array
            Target values
        """

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


            # Prediction

            y_pred = (
                X @ self.weights
                + self.bias
            )


            # Error

            error = (
                y_pred - y
            )


            # Gradient of weights
            #
            # MSE gradient:
            # (2/n) X.T(error)
            #
            # Ridge penalty gradient:
            # 2 lambda w

            dw = (
                (2 / n_samples)
                *
                (X.T @ error)
                +
                2 * self.lambda_
                *
                self.weights
            )


            # Gradient of bias
            #
            # Bias is NOT regularized

            db = (
                (2 / n_samples)
                *
                np.sum(error)
            )


            # Parameter update

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


            # Calculate loss

            mse = np.mean(
                error ** 2
            )


            ridge_penalty = (
                self.lambda_
                *
                np.sum(
                    self.weights ** 2
                )
            )


            loss = (
                mse
                +
                ridge_penalty
            )


            self.loss_history.append(
                loss
            )


    def predict(
        self,
        X
    ):
        """
        Generate predictions

        Parameters
        ----------
        X : numpy array

        Returns
        -------
        predictions : numpy array
        """

        return (
            X @ self.weights
            + self.bias
        )