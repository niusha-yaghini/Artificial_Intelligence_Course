import numpy as np


def initialize_pheromones(
    num_cities,
    initial_pheromone=1.0,
):
    """
    Initialize pheromone values equally
    for all city pairs.
    """

    return np.full(
        (
            num_cities,
            num_cities,
        ),
        initial_pheromone,
        dtype=float,
    )


def build_heuristic_matrix(
    problem,
):
    """
    Build heuristic information:

        eta(i, j) = 1 / distance(i, j)

    Only valid graph edges receive
    heuristic values.
    """

    heuristic = np.zeros_like(
        problem.distance_matrix,
        dtype=float,
    )

    valid_edges = (
        problem.adjacency_matrix
    )

    heuristic[
        valid_edges
    ] = (
        1.0
        /
        problem.distance_matrix[
            valid_edges
        ]
    )

    return heuristic


def calculate_transition_probabilities(
    current_city,
    available_cities,
    pheromones,
    heuristic,
    alpha=1.0,
    beta=2.0,
):
    """
    Calculate transition probabilities
    for available cities.

    P(i, j) is proportional to:

        pheromone^alpha
        *
        heuristic^beta
    """

    if len(
        available_cities
    ) == 0:
        return np.array([])


    pheromone_values = (
        pheromones[
            current_city,
            available_cities,
        ]
        ** alpha
    )


    heuristic_values = (
        heuristic[
            current_city,
            available_cities,
        ]
        ** beta
    )


    desirability = (
        pheromone_values
        *
        heuristic_values
    )


    total_desirability = np.sum(
        desirability
    )


    if total_desirability == 0:
        return np.full(
            len(available_cities),
            1.0
            /
            len(available_cities),
        )


    return (
        desirability
        /
        total_desirability
    )


def select_next_city(
    current_city,
    available_cities,
    pheromones,
    heuristic,
    alpha=1.0,
    beta=2.0,
):
    """
    Select the next city probabilistically.
    """

    if len(
        available_cities
    ) == 0:
        return None

    probabilities = (
        calculate_transition_probabilities(
            current_city,
            available_cities,
            pheromones,
            heuristic,
            alpha,
            beta,
        )
    )

    next_city = np.random.choice(
        available_cities,
        p=probabilities,
    )


    return int(
        next_city
    )


def construct_ant_route(
    problem,
    pheromones,
    heuristic,
    alpha=1.0,
    beta=2.0,
    start_city=None,
):
    """
    Construct one ant route.

    Returns:
        route
        is_feasible
    """

    # =====================================
    # Step 1:
    # Select starting city
    # =====================================

    if start_city is None:
        start_city = np.random.randint(
            problem.num_cities
        )

    route = [
        start_city
    ]

    visited = {
        start_city
    }

    current_city = start_city


    # =====================================
    # Step 2:
    # Visit remaining cities
    # =====================================

    while len(
        visited
    ) < problem.num_cities:

        neighbors = (
            problem.get_neighbors(
                current_city
            )
        )
        # print("neighbors: ", neighbors)


        available_cities = [
            city
            for city in neighbors
            if city not in visited
        ]

        # -------------------------------
        # Dead-end
        # -------------------------------

        if len(
            available_cities
        ) == 0:
            return (
                route,
                False,
            )

        # -------------------------------
        # Select next city
        # -------------------------------

        next_city = select_next_city(
            current_city,
            available_cities,
            pheromones,
            heuristic,
            alpha,
            beta,
        )

        route.append(
            next_city
        )

        visited.add(
            next_city
        )

        current_city = (
            next_city
        )

    # =====================================
    # Step 3:
    # Check return edge
    # =====================================

    if not problem.adjacency_matrix[
        current_city,
        start_city,
    ]:
        return (
            route,
            False,
        )

    # =====================================
    # Step 4:
    # Feasible route
    # =====================================

    return (
        route,
        True,
    )
    
# Note: This function modifies the matrix in-place and returns it.
def evaporate_pheromones(
    pheromones,
    evaporation_rate=0.2,
):
    """
    Apply pheromone evaporation.

    tau = (1 - rho) * tau
    """

    pheromones *= (
        1.0
        -
        evaporation_rate
    )

    return pheromones

def deposit_pheromones(
    pheromones,
    routes,
    route_lengths,
    q=1.0,
):
    """
    Deposit pheromone on edges used
    by feasible routes.

    Deposit amount:

        q / route_length
    """

    for route, route_length in zip(
        routes,
        route_lengths,
    ):

        if np.isinf(
            route_length
        ):
            continue


        deposit_amount = (
            q
            /
            route_length
        )


        full_route = (
            list(route)
            +
            [route[0]]
        )


        for i in range(
            len(full_route) - 1
        ):

            city_a = (
                full_route[i]
            )

            city_b = (
                full_route[i + 1]
            )


            pheromones[
                city_a,
                city_b
            ] += deposit_amount


            pheromones[
                city_b,
                city_a
            ] += deposit_amount


    return pheromones


def update_pheromones(
    pheromones,
    routes,
    route_lengths,
    evaporation_rate=0.2,
    q=1.0,
):
    """
    Complete pheromone update:
    evaporation + deposit.
    """

    evaporate_pheromones(
        pheromones,
        evaporation_rate,
    )

    deposit_pheromones(
        pheromones,
        routes,
        route_lengths,
        q,
    )

    return pheromones


def ant_colony_optimization(
    problem,
    num_ants=20,
    iterations=100,
    alpha=1.0,
    beta=2.0,
    evaporation_rate=0.2,
    q=1.0,
):
    """
    Complete Ant Colony Optimization algorithm
    for TSP.

    Returns:
        best_route
        best_length
        history
    """

    # =====================================
    # Step 1:
    # Initialize pheromone and heuristic
    # =====================================

    pheromones = initialize_pheromones(
        problem.num_cities
    )


    heuristic = build_heuristic_matrix(
        problem
    )


    # =====================================
    # Step 2:
    # Global best initialization
    # =====================================

    best_route = None

    best_length = np.inf


    history = []


    # =====================================
    # Step 3:
    # Main optimization loop
    # =====================================

    for iteration in range(
        iterations
    ):

        routes = []

        route_lengths = []


        # ---------------------------------
        # Each ant builds a solution
        # ---------------------------------

        for _ in range(
            num_ants
        ):

            route, feasible = (
                construct_ant_route(
                    problem,
                    pheromones,
                    heuristic,
                    alpha,
                    beta,
                )
            )


            if feasible:

                length = (
                    problem.route_length(
                        route
                    )
                )

            else:

                length = np.inf


            routes.append(
                route
            )

            route_lengths.append(
                length
            )


            # -----------------------------
            # Update global best
            # -----------------------------

            if length < best_length:

                best_length = length

                best_route = route.copy()


        # ---------------------------------
        # Update pheromones
        # ---------------------------------

        pheromones = update_pheromones(
            pheromones,
            routes,
            route_lengths,
            evaporation_rate,
            q,
        )


        # ---------------------------------
        # Save convergence history
        # ---------------------------------

        history.append(
            best_length
        )


    return {
        "route": best_route,
        "length": best_length,
        "history": history,
    }