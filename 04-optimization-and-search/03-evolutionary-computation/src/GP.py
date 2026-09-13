import random

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