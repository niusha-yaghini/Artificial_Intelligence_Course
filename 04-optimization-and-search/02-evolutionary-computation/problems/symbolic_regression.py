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

def hard_target_function(x):
    return (
        x**3
        +
        2*x**2
        -
        3*x
        +
        5
    )
    
def create_hard_dataset(
    start=-5,
    end=5,
    samples=100,
):
    x = np.linspace(
        start,
        end,
        samples
    )
    y = hard_target_function(
        x
    )

    return x, y

def trigonometric_function(x):
    return (
        np.sin(x)
        +
        0.5*np.cos(2*x)
    )
    
def create_trigonometric_dataset(
    start=-5,
    end=5,
    samples=100,
):
    x = np.linspace(
        start,
        end,
        samples
    )
    y = trigonometric_function(
        x
    )
    return x, y