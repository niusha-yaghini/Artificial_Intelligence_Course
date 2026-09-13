import numpy as np

def target_function(x):
    return (
        x**2
        +
        2*x
        +
        1
    )

def create_dataset(
    start=-5,
    end=5,
    samples=50,
):
    x = np.linspace(
        start,
        end,
        samples,
    )
    y = target_function(
        x
    )

    return x, y