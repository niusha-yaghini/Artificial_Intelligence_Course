import random
import numpy as np

def monte_carlo_pi(
    n_samples
):
    inside = 0

    for _ in range(n_samples):
        x = random.random()
        y = random.random()

        if x*x + y*y <= 1:
            inside += 1

    return 4 * inside / n_samples

def target(x):
    return np.exp(
        -x*x/2
    )
    
def proposal():
    return np.random.uniform(
        -3,
        3
    )
    
def rejection_sampling(
    n_samples,
    M=3
):
    samples=[]

    while len(samples)<n_samples:
        x = proposal()
        u = np.random.random()
        if u <= target(x)/M:
            samples.append(x)

    return np.array(samples)


class GibbsSampler:
    def __init__(
        self,
        p_a_given_b,
        p_b_given_a
    ):
        self.p_a_given_b = p_a_given_b
        self.p_b_given_a = p_b_given_a
        
    def sample(
        self,
        iterations=1000
    ):
        a,b = 0,0
        samples=[]

        for _ in range(iterations):
            p_a = self.p_a_given_b[b]
            a = (
                1
                if random.random() < p_a
                else 0
            )

            p_b = self.p_b_given_a[a]

            b = (
                1
                if random.random() < p_b
                else 0
            )

            samples.append(
                (a,b)
            )

        return samples