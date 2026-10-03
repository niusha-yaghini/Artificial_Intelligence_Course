import numpy as np


class SVM:

    def __init__(
        self,
        learning_rate=0.001,
        n_iterations=1000,
        lambda_param=0.01
    ):

        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param

        self.w = None
        self.b = None

    def fit(
        self,
        X,
        y
    ):

        y = np.where(
            y <= 0,
            -1,
            1
        )

        n_samples, n_features = X.shape

        self.w = np.zeros(
            n_features
        )

        self.b = 0


        for _ in range(
            self.n_iterations
        ):

            for idx, x_i in enumerate(X):

                condition = (
                    y[idx] *
                    (
                        np.dot(
                            x_i,
                            self.w
                        )
                        +
                        self.b
                    )
                    >= 1
                )


                if condition:

                    self.w -= (
                        self.lr *
                        2 *
                        self.lambda_param *
                        self.w
                    )


                else:

                    self.w -= (
                        self.lr *
                        (
                            2 *
                            self.lambda_param *
                            self.w
                            -
                            y[idx] *
                            x_i
                        )
                    )

                    self.b -= (
                        self.lr *
                        y[idx]
                        *
                        0.1
                    )

        return self


    def decision_function(
        self,
        X
    ):

        return (
            X @ self.w
            +
            self.b
        )


    def predict(
        self,
        X
    ):

        output = self.decision_function(
            X
        )

        return np.where(
            output >= 0,
            1,
            0
        )
        
    def get_margin(self):
        return 2 / np.linalg.norm(self.w)
    
    def get_support_vectors(
        self,
        X,
        y
    ):

        y_signed = np.where(
            y <= 0,
            -1,
            1
        )

        margins = y_signed * self.decision_function(
            X
        )

        return X[margins <= 1]