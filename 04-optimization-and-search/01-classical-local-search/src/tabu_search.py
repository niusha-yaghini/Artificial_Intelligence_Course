def binary_objective(
    state,
):
    return sum(state)

def get_neighbors(
    state,
):
    neighbors = []

    for i in range(len(state)):
        new_state = state.copy()
        new_state[i] = (
            1
            -
            new_state[i]
        )
        neighbors.append(
            (
                new_state,
                i
            )
        )

    return neighbors

def tabu_search(
    objective_function,
    initial_state,
    get_neighbors,
    iterations,
    tabu_tenure,
):
    current = initial_state
    best = current.copy()
    best_value = objective_function(
        best
    )

    tabu_list = []

    for _ in range(iterations):
        neighbors = get_neighbors(
            current
        )

        candidate = None
        candidate_move = None
        candidate_value = float("-inf")

        for neighbor, move in neighbors:
            value = objective_function(
                neighbor
            )
            is_tabu = (
                move
                in
                tabu_list
            )
            if is_tabu and value <= best_value:
                continue
            if value > candidate_value:
                candidate = neighbor
                candidate_move = move
                candidate_value = value

        # if candidate is None:
        #     break
        
        if candidate is None:
            aspiration_candidates = [
                (neighbor, move)
                for neighbor, move in neighbors
                if objective_function(neighbor) > best_value
            ]
            if aspiration_candidates:
                candidate, candidate_move = max(
                    aspiration_candidates,
                    key=lambda item: objective_function(item[0])
                )
            else:
                break

        current = candidate

        tabu_list.append(
            candidate_move
        )

        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        if candidate_value > best_value:
            best = current.copy()
            best_value = candidate_value

    return best

def tabu_search_with_history(
    objective_function,
    initial_state,
    get_neighbors,
    iterations,
    tabu_tenure,
):
    current = initial_state.copy()
    best = current.copy()
    best_value = objective_function(
        best
    )

    tabu_list = []
    history = []

    for iteration in range(iterations):
        neighbors = get_neighbors(
            current
        )

        candidate = None
        candidate_move = None
        candidate_value = float("-inf")

        for neighbor, move in neighbors:
            value = objective_function(
                neighbor
            )
            is_tabu = (
                move in tabu_list
            )
            if is_tabu and value <= best_value:
                continue
            if value > candidate_value:
                candidate = neighbor
                candidate_move = move
                candidate_value = value

        if candidate is None:
            break

        current = candidate

        tabu_list.append(
            candidate_move
        )

        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        if candidate_value > best_value:
            best = current.copy()
            best_value = candidate_value

        history.append(
            {
                "iteration": iteration,
                "current_value": candidate_value,
                "best_value": best_value,
                "tabu_list": tabu_list.copy(),
            }
        )

    return best, history