import random
import numpy as np
import copy

class Node:
    def __init__(
        self,
        value,
        left=None,
        right=None,
    ):
        # Node value:
        # operator -> "+", "-", "*", "/"
        # terminal -> "x", number
        self.value = value

        # Children nodes
        self.left = left
        self.right = right

    # Evaluate Expression Tree
    def evaluate(
        self,
        x,
    ):
        # Terminal Nodes

        # Variable x
        if self.value == "x":
            return x

        # Constant values
        elif isinstance(
            self.value,
            (int, float)
        ):
            return self.value

        # Function Nodes
        elif self.value == "+":
            return (
                self.left.evaluate(x)
                +
                self.right.evaluate(x)
            )
        elif self.value == "-":
            return (
                self.left.evaluate(x)
                -
                self.right.evaluate(x)
            )
        elif self.value == "*":
            return (
                self.left.evaluate(x)
                *
                self.right.evaluate(x)
            )
        elif self.value == "/":
            denominator = self.right.evaluate(x)
            # Protected Division
            if denominator == 0:
                return 1
            return (
                self.left.evaluate(x)
                /
                denominator
            )
        else:
            raise ValueError(
                f"Unknown node value: {self.value}"
            )

    # Convert Tree to Mathematical Expression
    def to_string(
        self,
    ):
        # Terminal node
        if self.left is None and self.right is None:
            return str(
                self.value
            )

        # Operator node
        return (
            "("
            +
            self.left.to_string()
            +
            f" {self.value} "
            +
            self.right.to_string()
            +
            ")"
        )
    
    def get_nodes(
        self,
    ):
        nodes = [self]
        if self.left is not None:
            nodes.extend(
                self.left.get_nodes()
            )
        if self.right is not None:
            nodes.extend(
                self.right.get_nodes()
            )
        return nodes
    
    def copy_tree(
        self,
    ):
        return copy.deepcopy(
            self
        )
        
        
FUNCTIONS = [
    "+",
    "-",
    "*",
    "/",
]

TERMINALS = [
    "x",
    1,
    2,
    3,
]

def generate_random_tree(
    max_depth,
    current_depth=0,
):
    # Stop condition: If maximum depth reached, create terminal node
    if current_depth >= max_depth:
        return Node(
            random.choice(
                TERMINALS
            )
        )

    # Decide node type
    if random.random() < 0.7:
        # Function Node
        function = random.choice(
            FUNCTIONS
        )
        return Node(
            function,
            generate_random_tree(
                max_depth,
                current_depth + 1,
            ),
            generate_random_tree(
                max_depth,
                current_depth + 1,
            ),
        )
    else:
        # Terminal Node
        return Node(
            random.choice(
                TERMINALS
            )
        )


def fitness_function(
    tree,
    x_data,
    y_data,
):
    predictions = []

    for x in x_data:
        try:
            value = tree.evaluate(
                x
            )
        except:
            value = 1e6
        predictions.append(
            value
        )

    predictions = np.array(
        predictions
    )

    error = np.mean(
        (predictions - y_data)**2
    )

    return error


def tournament_selection(
    population,
    fitness_values,
    tournament_size=3,
):
    # Step 1: Randomly choose individuals
    participants = random.sample(
        range(len(population)),
        tournament_size,
    )

    # Step 2: Find best individual (Lower fitness is better)
    best_index = participants[0]

    for index in participants[1:]:
        if fitness_values[index] < fitness_values[best_index]:
            best_index = index

    # Step 3: Return selected Tree
    return population[best_index]


def get_random_node(
    tree,
):
    nodes = tree.get_nodes()
    return random.choice(
        nodes
    )
    

def subtree_crossover(
    parent1,
    parent2,
):
    # Copy parents
    child = parent1.copy_tree()

    # Select random subtree from parent2
    donor_subtree = get_random_node(
        parent2
    )

    # Select random node in child
    target_node = get_random_node(
        child
    )

    # Replace target value
    target_node.value = donor_subtree.value
    target_node.left = donor_subtree.left
    target_node.right = donor_subtree.right

    return child


def subtree_mutation(
    tree,
    max_depth=3,
):
    # Step 1: Create independent copy
    mutated_tree = copy.deepcopy(
        tree
    )

    # Step 2: Select random node
    mutation_node = get_random_node(
        mutated_tree
    )

    # Step 3: Generate new random subtree
    new_subtree = generate_random_tree(
        max_depth
    )

    # Step 4: Replace selected node
    mutation_node.value = new_subtree.value
    mutation_node.left = new_subtree.left
    mutation_node.right = new_subtree.right

    return mutated_tree


def genetic_programming(
    x_data,
    y_data,
    population_size=100,
    generations=50,
    max_depth=3,
    crossover_rate=0.9,
    mutation_rate=0.1,
):
    population = []

    # Step 1: Initialize Population
    for _ in range(population_size):
        population.append(
            generate_random_tree(
                max_depth
            )
        )

    history = []
    best_tree = None
    best_fitness = float("inf")

    # Step 2: Evolution Loop
    for generation in range(
        generations
    ):
        fitness_values = []

        # Evaluate Population
        for tree in population:
            fitness_values.append(
                fitness_function(
                    tree,
                    x_data,
                    y_data,
                )
            )

        # Find Best
        best_index = np.argmin(
            fitness_values
        )

        if fitness_values[best_index] < best_fitness:
            best_fitness = fitness_values[
                best_index
            ]
            best_tree = copy.deepcopy(
                population[best_index]
            )

        history.append(
            best_fitness
        )

        # Step 3: Create Next Generation
        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_selection(
                population,
                fitness_values,
            )
            parent2 = tournament_selection(
                population,
                fitness_values,
            )

            if random.random() < crossover_rate:
                child = subtree_crossover(
                    parent1,
                    parent2,
                )
            else:
                child = copy.deepcopy(
                    parent1
                )

            if random.random() < mutation_rate:
                child = subtree_mutation(
                    child,
                    max_depth,
                )

            new_population.append(
                child
            )

        population = new_population

    return {
        "best_tree": best_tree,
        "fitness": best_fitness,
        "history": history,
    }