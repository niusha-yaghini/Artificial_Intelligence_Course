import math
import random


def acceptance_probability(
    current_value,
    new_value,
    temperature,
):
    if new_value > current_value:
        return 1.0
    difference = (
        new_value -
        current_value
    )
    return math.exp(
        difference / temperature
    )
    
def simulated_annealing(
    objective_function,
    initial_state,
    get_neighbor,
    initial_temperature,
    cooling_rate,
    minimum_temperature,
):
    current = initial_state
    temperature = initial_temperature

    while temperature > minimum_temperature:
        neighbor = get_neighbor(
            current
        )
        current_value = objective_function(
            current
        )
        neighbor_value = objective_function(
            neighbor
        )
        probability = acceptance_probability(
            current_value,
            neighbor_value,
            temperature,
        )
        if random.random() < probability:
            current = neighbor
        temperature *= cooling_rate

    return current

def simulated_annealing_with_best(
    objective_function,
    initial_state,
    get_neighbor,
    initial_temperature,
    cooling_rate,
    minimum_temperature,
):
    current = initial_state
    current_value = objective_function(
        current
    )

    best = current
    best_value = current_value
    temperature = initial_temperature

    while temperature > minimum_temperature:
        neighbor = get_neighbor(
            current
        )
        neighbor_value = objective_function(
            neighbor
        )
        probability = acceptance_probability(
            current_value,
            neighbor_value,
            temperature,
        )
        if random.random() < probability:
            current = neighbor
            current_value = neighbor_value
        if current_value > best_value:
            best = current
            best_value = current_value

        temperature *= cooling_rate

    return best

# def random_neighbor(x):
#     step = random.choice(
#         [-1,1]
#     )
#     return x + step

def random_neighbor(x):
    return x + random.uniform(
        -1,
        1
    )
    
def bounded_random_neighbor(
    x,
    lower_bound=0,
    upper_bound=20,
    step_size=1.0,
):
    neighbor = x + random.uniform(
        -step_size,
        step_size,
    )

    return max(
        lower_bound,
        min(neighbor, upper_bound),
    )
    
def simulated_annealing_with_history(
    objective_function,
    initial_state,
    get_neighbor,
    initial_temperature,
    cooling_rate,
    minimum_temperature,
):
    current = initial_state
    current_value = objective_function(
        current
    )

    best = current
    best_value = current_value
    temperature = initial_temperature
    history = []

    while temperature > minimum_temperature:
        neighbor = get_neighbor(
            current
        )
        neighbor_value = objective_function(
            neighbor
        )
        probability = acceptance_probability(
            current_value,
            neighbor_value,
            temperature,
        )
        if random.random() < probability:
            current = neighbor
            current_value = neighbor_value
        if current_value > best_value:
            best = current
            best_value = current_value

        history.append(
            {
                "temperature": temperature,
                "current_value": current_value,
                "best_value": best_value,
            }
        )

        temperature *= cooling_rate

    return best, history