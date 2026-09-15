import numpy as np

def sphere_function(x):
    return np.sum(
        x**2
    )

def rosenbrock_function(x):
    return np.sum(
        100*
        (
            x[1:]
            -
            x[:-1]**2
        )**2
        +
        (
            1-x[:-1]
        )**2
    )

def rastrigin_function(x):
    n = len(x)
    return (
        10*n
        +
        np.sum(
            x**2
            -
            10*np.cos(
                2*np.pi*x
            )
        )
    )
    
def ackley_function(x):
    n = len(x)
    term1 = (
        -20
        *
        np.exp(
            -0.2
            *
            np.sqrt(
                np.sum(x**2)
                /
                n
            )
        )
    )
    term2 = (
        -np.exp(
            np.sum(
                np.cos(
                    2*np.pi*x
                )
            )
            /
            n
        )
    )
    return (
        term1
        +
        term2
        +
        20
        +
        np.e
    )