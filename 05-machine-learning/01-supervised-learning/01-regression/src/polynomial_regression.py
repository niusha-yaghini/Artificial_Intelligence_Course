import numpy as np

class PolynomialFeatures:
    def __init__(
        self,
        degree=2
    ):
        self.degree = degree
        
    def transform(
        self,
        X
    ):

        X_poly = []

        for x in X:

            x = x[0]

            features = []

            for d in range(
                1,
                self.degree + 1
            ):

                features.append(
                    x ** d
                )


            X_poly.append(
                features
            )


        return np.array(
            X_poly
        )