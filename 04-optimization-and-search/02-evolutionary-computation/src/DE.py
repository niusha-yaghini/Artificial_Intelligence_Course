import numpy as np
import random


def initialize_population(
    population_size,
    dimension,
    bounds,
):
    population = []

    for _ in range(population_size):
        individual = [
            random.uniform(
                bounds[i][0],
                bounds[i][1],
            )
            for i in range(dimension)
        ]
        population.append(
            individual
        )

    return np.array(
        population
    )
    
def mutation(
    population,
    target_index,
    F=0.5,
):
    population_size = len(
        population
    )
    indices = list(
        range(population_size)
    )
    indices.remove(
        target_index
    )
    r1, r2, r3 = random.sample(
        indices,
        3,
    )

    x1 = population[r1]
    x2 = population[r2]
    x3 = population[r3]

    mutant = (
        x1
        +
        F * (x2 - x3)
    )

    return mutant

def boundary_handling(
    vector,
    bounds,
):
    vector = vector.copy()

    for i in range(
        len(vector)
    ):
        lower_bound, upper_bound = bounds[i]

        if vector[i] < lower_bound:
            vector[i] = lower_bound

        elif vector[i] > upper_bound:
            vector[i] = upper_bound

    return vector

def mutation_with_boundary(
    population,
    target_index,
    F=0.5,
    bounds=None,
):
    # Step 1:
    # Create mutant vector
    mutant = mutation(
        population,
        target_index,
        F,
    )

    # Step 2:
    # Apply boundary constraints
    if bounds is not None:
        mutant = boundary_handling(
            mutant,
            bounds,
        )

    return mutant

def crossover(
    target,
    mutant,
    CR=0.5,
):
    dimension = len(
        target
    )

    trial = target.copy()

    # Ensure at least one mutant component is selected
    j_rand = random.randint(
        0,
        dimension - 1
    )

    for j in range(
        dimension
    ):
        if (
            random.random() < CR
            or
            j == j_rand
        ):
            trial[j] = mutant[j]

    return trial

def selection(
    target,
    trial,
    objective_function,
):
    target_fitness = objective_function(
        target
    )
    trial_fitness = objective_function(
        trial
    )

    if trial_fitness < target_fitness:
        return trial
    else:
        return target
    
def differential_evolution(
    objective_function,
    bounds,
    population_size=50,
    generations=100,
    F=0.5,
    CR=0.7,
):
    # Step 1: Initialization
    # Determine problem dimension and create initial population
    dimension = len(bounds)

    population = initialize_population(
        population_size,
        dimension,
        bounds,
    )

    # Step 2: Store Best Solution Information
    # Since DE is usually used for minimization:
    # lower fitness = better solution
    best_solution = None
    best_fitness = float(
        "inf"
    )

    # Store convergence history
    history = []
    best_solution_history = []

    # Step 3: Evolution Loop
    # Repeat the optimization process
    for generation in range(
        generations
    ):
        # Step 4: Generate New Candidates
        # Each individual competes with its own trial vector
        for i in range(
            population_size
        ):
            # Step 4.1: Mutation
            # Create mutant vector using DE/rand/1
            # V = Xr1 + F(Xr2 - Xr3)
            mutant = mutation_with_boundary(
                population,
                i,
                F,
                bounds,
            )

            # Step 4.2: Crossover
            # Combine target vector and mutant vector
            # Create trial vector
            trial = crossover(
                population[i],
                mutant,
                CR,
            )

            # Step 4.3: Boundary Handling
            # Make sure trial vector remains feasible
            trial = boundary_handling(
                trial,
                bounds,
            )

            # Step 4.4: Selection
            # Trial competes against target vector
            population[i] = selection(
                population[i],
                trial,
                objective_function,
            )

        # Step 5: Evaluate Current Population
        # Find best individual of current generation
        fitness_values = [
            objective_function(individual)
            for individual in population
        ]

        best_index = np.argmin(
            fitness_values
        )

        current_best_fitness = fitness_values[
            best_index
        ]

        # Update global best solution
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[
                best_index
            ].copy()
        
        # Save convergence history
        history.append(
            best_fitness
        )
        best_solution_history.append(
            best_solution.copy()
        )

    # Step 6: Return Results
    return {
        "solution": best_solution,
        "fitness": best_fitness,
        "history": history,
        "solution_history": best_solution_history,
    }