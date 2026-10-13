import random


def initialize_population(
    population_size,
    chromosome_length,
):
    population = []

    for _ in range(population_size):
        chromosome = [
            random.randint(0,1)
            for _ in range(chromosome_length)
        ]
        population.append(
            chromosome
        )

    return population

def evaluate_population(
    population,
    problem,
):
    fitness_values = []

    for chromosome in population:
        fitness = problem.evaluate(
            chromosome
        )
        fitness_values.append(
            fitness
        )

    return fitness_values

def tournament_selection(
    population,
    fitness_values,
    tournament_size,
):
    tournament_indices = random.sample(
        range(len(population)),
        tournament_size,
    )
    winner_index = max(
        tournament_indices,
        key=lambda index:
        fitness_values[index],
    )
    
    return population[winner_index]

def elitism(
    population,
    fitness_values,
    elite_ratio,
):
    elite_count = int(
        len(population)
        *
        elite_ratio
    )
    ranked_indices = sorted(
        range(len(population)),
        key=lambda i: fitness_values[i],
        reverse=True,
    )
    elites = [
        population[i]
        for i in ranked_indices[:elite_count]
    ]

    return elites

def roulette_selection(
    population,
    fitness_values,
):
    total_fitness = sum(
        fitness_values
    )
    probabilities = [
        fitness / total_fitness
        for fitness in fitness_values
    ]
    selected_index = random.choices(
        range(len(population)),
        weights=probabilities,
        k=1,
    )[0]

    return population[selected_index]

def single_point_crossover(
    parent1,
    parent2,
    crossover_rate=0.8,
    crossover_point_ratio=0.5,
):
    if random.random() > crossover_rate:
        return (
            parent1.copy(),
            parent2.copy(),
        )

    point = int(
        len(parent1)
        *
        crossover_point_ratio
    )
    child1 = (
        parent1[:point]
        +
        parent2[point:]
    )
    child2 = (
        parent2[:point]
        +
        parent1[point:]
    )

    return child1, child2

def mutation(
    chromosome,
    mutation_rate=0.01,
):
    mutated = chromosome.copy()

    for i in range(
        len(mutated)
    ):
        if random.random() < mutation_rate:
            mutated[i] = (
                1
                -
                mutated[i]
            )

    return mutated


def genetic_algorithm(
    problem,
    chromosome_length,
    population_size=100,
    generations=100,
    elite_ratio=0.1,
    selection_method="tournament",
    tournament_size=3,
    crossover_rate=0.8,
    crossover_point_ratio=0.5,
    mutation_rate=0.01,
):
    # Step 1: Initialize Population
    # Create the first generation randomly
    population = initialize_population(
        population_size,
        chromosome_length,
    )

    # Store best solution information
    best_solution = None
    best_fitness = float("-inf")

    # Store fitness progress for visualization
    fitness_history = []

    # Step 2: Evolution Loop
    # Repeat the process for multiple generations
    for generation in range(generations):
        # Step 3: Evaluate Population Fitness
        # Calculate fitness value for every chromosome
        fitness_values = evaluate_population(
            population,
            problem,
        )

        # Step 4: Find Best Individual of Current Generation
        # Keep track of the best solution found so far
        current_best_index = max(
            range(len(population)),
            key=lambda i: fitness_values[i],
        )

        current_best_fitness = fitness_values[
            current_best_index
        ]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[
                current_best_index
            ].copy()

        # Save convergence history
        fitness_history.append(
            best_fitness
        )

        # Step 5: Elitism
        # Directly preserve the best individuals
        elites = elitism(
            population,
            fitness_values,
            elite_ratio,
        )

        # Number of offspring needed
        offspring_needed = (
            population_size
            -
            len(elites)
        )

        offspring = []

        # Step 6: Selection + Crossover
        # Select parent pairs and generate offspring
        # until the next generation is complete
        while len(offspring) < offspring_needed:
            # Parent Selection
            if selection_method == "tournament":
                parent1 = tournament_selection(
                    population,
                    fitness_values,
                    tournament_size,
                )
                parent2 = tournament_selection(
                    population,
                    fitness_values,
                    tournament_size,
                )
            elif selection_method == "roulette":
                parent1 = roulette_selection(
                    population,
                    fitness_values,
                )
                parent2 = roulette_selection(
                    population,
                    fitness_values,
                )
            else:
                raise ValueError(
                    "Unknown selection method"
                )

            # Crossover
            child1, child2 = single_point_crossover(
                parent1,
                parent2,
                crossover_rate,
                crossover_point_ratio,
            )

            offspring.extend(
                [
                    child1,
                    child2,
                ]
            )

        # Remove extra children if population size is odd
        offspring = offspring[
            :offspring_needed
        ]

        # Step 7: Mutation
        # Apply mutation after all offspring
        # are generated
        offspring = [
            mutation(
                child,
                mutation_rate,
            )

            for child in offspring
        ]

        # Step 8: Create New Generation
        # Elite individuals + mutated offspring
        population = (
            elites
            +
            offspring
        )

    # Step 9: Return Final Result
    return {
        "solution": best_solution,
        "fitness": best_fitness,
        "history": fitness_history,
    }