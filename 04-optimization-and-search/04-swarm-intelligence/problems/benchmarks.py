# Ackley Function

import numpy as np

def sphere_function(x):
    return np.sum(
        np.square(x)
    )

def rosenbrock_function(x):
    return (
        (1 - x[0]) ** 2
        +
        100 * (x[1] - x[0] ** 2) ** 2
    )

def rastrigin_function(x):
    n = len(x)
    return (
        10 * n
        +
        np.sum(
            x ** 2
            -
            10 * np.cos(2 * np.pi * x)
        )
    )