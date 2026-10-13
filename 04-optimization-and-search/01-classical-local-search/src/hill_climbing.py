import random

def hill_climbing(
    objective_function,
    initial_state,
    get_neighbors,
):
    current = initial_state

    while True:
        neighbors = get_neighbors(
            current
        )
        current_value = objective_function(
            current
        )
        neighbor_values = [
            (
                neighbor,
                objective_function(neighbor)
            )
            for neighbor in neighbors
        ]
        best_neighbor, best_value = max(
            neighbor_values,
            key=lambda item:item[1]
        )

        if best_value <= current_value:
            break

        current = best_neighbor

    return current

def random_restart_hill_climbing(
    objective_function,
    get_neighbors,
    search_range,
    restarts=10,
):
    best_solution = None
    best_value = float("-inf")

    for _ in range(restarts):
        initial_state = random.uniform(
            search_range[0],
            search_range[1],
        )
        solution = hill_climbing(
            objective_function,
            initial_state,
            get_neighbors,
        )
        value = objective_function(
            solution
        )
        if value > best_value:
            best_value = value
            best_solution = solution

    return best_solution

def get_neighbors(
    x,
    step_size=1
):
    return [
        x - step_size,
        x + step_size,
    ]