import random


class KnapsackProblem:
    def __init__(
        self,
        weights,
        values,
        capacity,
    ):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        
    def evaluate(
        self,
        chromosome,
    ):
        total_weight = 0
        total_value = 0

        for gene, weight, value in zip(
            chromosome,
            self.weights,
            self.values,
        ):
            if gene == 1:
                total_weight += weight
                total_value += value

        if total_weight > self.capacity:
            return 0

        return total_value
    
def create_random_knapsack(
    n_items=100,
    seed=42,
):
    random.seed(seed)

    weights = [
        random.randint(1,50)
        for _ in range(n_items)
    ]
    values = [
        random.randint(10,100)
        for _ in range(n_items)
    ]
    capacity = int(
        sum(weights)
        *
        0.4
    )
    return (
        weights,
        values,
        capacity,
    )