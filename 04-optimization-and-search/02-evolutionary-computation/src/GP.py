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
    
    
# ------------------------------------------------------
# Advanced Version
# ------------------------------------------------------

class NodeAdvanced:
    def __init__(
        self,
        value,
        children=None,
    ):
        # Node value:
        self.value = value

        # Children list
        if children is None:
            self.children = []
        else:
            self.children = children
            
    def evaluate(
        self,
        x,
    ):
        # Terminal Nodes
        if self.value == "x":
            return x
        elif isinstance(
            self.value,
            (int,float)
        ):
            return self.value
        # Binary Operators
        elif self.value == "+":
            return (
                self.children[0].evaluate(x)
                +
                self.children[1].evaluate(x)
            )
        elif self.value == "-":
            return (
                self.children[0].evaluate(x)
                -
                self.children[1].evaluate(x)
            )
        elif self.value == "*":
            return (
                self.children[0].evaluate(x)
                *
                self.children[1].evaluate(x)
            )
        elif self.value == "/":
            denominator = self.children[1].evaluate(x)
            if abs(denominator) < 1e-8:
                return 1
            return (
                self.children[0].evaluate(x)
                /
                denominator
            )
        # Unary Operators
        elif self.value == "sin":
            return np.sin(
                self.children[0].evaluate(x)
            )
        elif self.value == "cos":
            return np.cos(
                self.children[0].evaluate(x)
            )
        # elif self.value == "exp":
        #     value = self.children[0].evaluate(x)
        #     return np.exp(
        #         np.clip(
        #             value,
        #             -10,
        #             10
        #         )
        #     )
        # elif self.value == "log":
        #     value = self.children[0].evaluate(x)
        #     return np.log(
        #         abs(value)+1e-8
        #     )
        else:
            raise ValueError(
                f"Unknown operator: {self.value}"
            )
            
    # Convert Tree to Mathematical Expression
    def to_string(self):
        if len(self.children) == 0:
            return str(
                self.value
            )
        elif len(self.children) == 1:
            return (
                f"{self.value}"
                "("
                +
                self.children[0].to_string()
                +
                ")"
            )
        elif len(self.children) == 2:
            return (
                "("
                +
                self.children[0].to_string()
                +
                f" {self.value} "
                +
                self.children[1].to_string()
                +
                ")"
            )
    
    # Get All Nodes
    def get_nodes(self):
        nodes = [
            self
        ]
        for child in self.children:
            nodes.extend(
                child.get_nodes()
            )
        return nodes
    
    # Deep Copy Tree
    def copy_tree(self):
        return copy.deepcopy(
            self
        )
        
    # Count Number of Nodes
    def tree_size(self):
        size = 1
        for child in self.children:
            size += child.tree_size()
        return size
    
    # Here, we want to create a dictionary that maps each node to its parent.
    def get_parent_map(
        self,
        mapping=None,
    ):
        if mapping is None:
            mapping = {}

        for child in self.children:
            mapping[child] = self
            child.get_parent_map(
                mapping
            )

        return mapping

        
ADVANCED_FUNCTIONS = [
    "+",
    "-",
    "*",
    "/",
    "sin",
    "cos",
    # "exp",
    # "log",
]

def generate_random_constant():
    value = random.uniform(-5, 5)
    return round(value, 2)
    
def generate_terminal_advanced():
    # 70% variable x
    if random.random() < 0.7:
        return NodeAdvanced(
            "x"
        )
    # 30% random constant
    else:
        return NodeAdvanced(
            generate_random_constant()
        )
        
def get_function_arity(
    function
):
    if function in [
        "+",
        "-",
        "*",
        "/",
    ]:
        return 2
    elif function in [
        "sin",
        "cos",
        # "exp",
        # "log",
    ]:
        return 1
    

def generate_random_tree_advanced(
    max_depth,
    current_depth=0,
):
    # Stop Condition
    if current_depth >= max_depth:
        return generate_terminal_advanced()

    # Decide Terminal or Function
    if random.random() < 0.8:
        function = random.choice(
            ADVANCED_FUNCTIONS
        )
        arity = get_function_arity(
            function
        )

        children = []

        for _ in range(arity):
            children.append(
                generate_random_tree_advanced(
                    max_depth,
                    current_depth + 1
                )
            )
        return NodeAdvanced(
            function,
            children
        )
    else:
        return generate_terminal_advanced()
    

def advanced_fitness_function(
    tree,
    x_data,
    y_data,
    complexity_weight=0.001,
):
    predictions = []

    # Evaluate Tree
    for x in x_data:
        try:
            value = tree.evaluate(x)
            # Check invalid values
            if (
                np.isnan(value)
                or
                np.isinf(value)
            ):
                return 1e6
        except Exception:
            return 1e6

        predictions.append(
            value
        )

    predictions = np.array(
        predictions
    )

    # Prediction Error
    mse = np.mean(
        (predictions - y_data)**2
    )

    # Complexity Penalty
    complexity_penalty = (
        complexity_weight
        *
        tree.tree_size()
    )

    # Final Fitness
    fitness = (
        mse
        +
        complexity_penalty
    )

    return fitness


def subtree_crossover_advanced(
    parent1,
    parent2,
):
    # Step 1: Copy Parent
    child = parent1.copy_tree()

    # Step 2: Select donor subtree
    donor_nodes = parent2.get_nodes()
    donor = copy.deepcopy(
        random.choice(
            donor_nodes
        )
    )

    # Step 3: Select target node
    target_nodes = child.get_nodes()
    target = random.choice(
        target_nodes
    )

    # Step 4: Replace target
    parent_map = child.get_parent_map()

    # If the target is the root
    if target == child:
        child = donor
    else:
        parent = parent_map[target]
        index = parent.children.index(
            target
        )
        parent.children[index] = donor

    return child


def subtree_mutation_advanced(
    tree,
    max_depth=3,
):
    # Step 1: Copy Tree
    mutated = tree.copy_tree()

    # Step 2: Select node
    target_nodes = mutated.get_nodes()
    target = random.choice(
        target_nodes
    )

    # Step 3: Generate replacement
    new_subtree = generate_random_tree_advanced(
        max_depth
    )

    # Step 4: Replace
    parent_map = mutated.get_parent_map()

    if target == mutated:
        mutated = new_subtree
    else:
        parent = parent_map[target]
        index = parent.children.index(
            target
        )
        parent.children[index] = new_subtree

    return mutated


def genetic_programming_advanced(
    x_data,
    y_data,
    population_size=100,
    generations=50,
    max_depth=4,
    crossover_rate=0.9,
    mutation_rate=0.1,
):
    # Step 1: Initialize Population
    population = []

    for _ in range(
        population_size
    ):
        population.append(
            generate_random_tree_advanced(
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
            fitness = advanced_fitness_function(
                tree,
                x_data,
                y_data,
            )
            fitness_values.append(
                fitness
            )

        # Find Best
        best_index = np.argmin(
            fitness_values
        )
        current_best = fitness_values[
            best_index
        ]

        if current_best < best_fitness:
            best_fitness = current_best
            best_tree = copy.deepcopy(
                population[best_index]
            )

        history.append(
            best_fitness
        )

        # Step 3: Create New Generation
        new_population = []

        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(
                population,
                fitness_values,
            )
            parent2 = tournament_selection(
                population,
                fitness_values,
            )

            # Crossover
            if random.random() < crossover_rate:
                child = subtree_crossover_advanced(
                    parent1,
                    parent2,
                )
            else:
                child = copy.deepcopy(
                    parent1
                )

            # Mutation
            if random.random() < mutation_rate:
                child = subtree_mutation_advanced(
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