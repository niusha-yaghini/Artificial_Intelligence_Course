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