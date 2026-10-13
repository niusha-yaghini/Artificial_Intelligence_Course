import numpy as np


def create_complete_graph(
    num_cities,
):
    """
    Create a complete undirected graph.

    Every city is connected to every other city.
    Self-connections are disabled.
    """

    adjacency_matrix = np.ones(
        (
            num_cities,
            num_cities,
        ),
        dtype=bool,
    )

    np.fill_diagonal(
        adjacency_matrix,
        False,
    )

    return adjacency_matrix


def create_random_graph(
    num_cities,
    edge_probability=0.5,
    seed=None,
):
    """
    Create a random undirected graph.

    Each possible edge is added independently
    with the given probability.
    """

    rng = np.random.default_rng(
        seed
    )

    adjacency_matrix = np.zeros(
        (
            num_cities,
            num_cities,
        ),
        dtype=bool,
    )

    for i in range(
        num_cities
    ):
        for j in range(
            i + 1,
            num_cities,
        ):
            if (
                rng.random()
                <
                edge_probability
            ):
                adjacency_matrix[
                    i,
                    j
                ] = True

                adjacency_matrix[
                    j,
                    i
                ] = True

    return adjacency_matrix


class TSPProblem:

    def __init__(
        self,
        cities,
        adjacency_matrix,
    ):

        self.cities = np.array(
            cities,
            dtype=float,
        )

        self.num_cities = len(
            self.cities
        )

        self.adjacency_matrix = np.array(
            adjacency_matrix,
            dtype=bool,
        )

        self._validate_graph()

        self.distance_matrix = (
            self._build_distance_matrix()
        )


    def _validate_graph(self):
        """
        Validate the basic structure of the graph.
        """

        expected_shape = (
            self.num_cities,
            self.num_cities,
        )

        if (
            self.adjacency_matrix.shape
            != expected_shape
        ):
            raise ValueError(
                "Adjacency matrix shape "
                "must match number of cities."
            )

        if np.any(
            np.diag(
                self.adjacency_matrix
            )
        ):
            raise ValueError(
                "Self-connections are not allowed."
            )


    def _build_distance_matrix(self):
        """
        Build the distance matrix.

        Valid edges receive Euclidean distance.
        Invalid edges receive infinity.
        """

        matrix = np.full(
            (
                self.num_cities,
                self.num_cities,
            ),
            np.inf,
            dtype=float,
        )

        np.fill_diagonal(
            matrix,
            0.0,
        )

        for i in range(
            self.num_cities
        ):
            for j in range(
                self.num_cities
            ):
                if not self.adjacency_matrix[
                    i,
                    j
                ]:
                    continue

                matrix[
                    i,
                    j
                ] = np.linalg.norm(
                    self.cities[i]
                    -
                    self.cities[j]
                )

        return matrix


    def get_neighbors(
        self,
        city,
    ):
        """
        Return all cities directly connected
        to the given city.
        """

        return np.where(
            self.adjacency_matrix[
                city
            ]
        )[0].tolist()


    def route_length(
        self,
        route,
    ):
        """
        Calculate the total route length.

        The route automatically returns
        to the starting city.

        If any required edge does not exist,
        return infinity.
        """

        total_distance = 0.0

        full_route = (
            list(route)
            +
            [route[0]]
        )

        for i in range(
            len(full_route) - 1
        ):
            current_city = (
                full_route[i]
            )

            next_city = (
                full_route[i + 1]
            )

            distance = (
                self.distance_matrix[
                    current_city,
                    next_city,
                ]
            )

            if np.isinf(
                distance
            ):
                return np.inf

            total_distance += (
                distance
            )

        return total_distance