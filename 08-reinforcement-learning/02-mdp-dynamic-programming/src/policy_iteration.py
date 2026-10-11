import numpy as np

def policy_evaluation(
    env,
    n_states,
    policy,
    V,
    gamma=0.99,
    theta=1e-8,
):

    while True:
        delta = 0
        for state in range(n_states):
            value = 0
            for action, action_prob in enumerate(
                policy[state]
            ):
                for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                    value += (
                        action_prob
                        *
                        prob
                        *
                        (
                            reward
                            +
                            gamma * V[next_state]
                        )
                    )

            delta = max(
                delta,
                abs(
                    V[state]-value
                )
            )

            V[state] = value

        if delta < theta:
            break

    return V


def calculate_q_value(
    env,
    state,
    action,
    V,
    gamma=0.99
):

    q = 0

    for prob, next_state, reward, done in env.unwrapped.P[state][action]:

        q += prob * (
            reward
            +
            gamma * V[next_state]
        )

    return q


def policy_improvement(
    env,
    n_states,
    n_actions,
    V,
    gamma=0.99
):

    new_policy = np.zeros(
        (
            n_states,
            n_actions
        )
    )

    for state in range(n_states):

        q_values = []

        for action in range(n_actions):

            q_values.append(
                calculate_q_value(
                    env,
                    state,
                    action,
                    V,
                    gamma
                )
            )

        best_action = np.argmax(
            q_values
        )

        new_policy[state][best_action] = 1

    return new_policy


def policy_iteration(
    env,
    n_states,
    n_actions,
    gamma=0.99
):
    policy = np.ones(
        (
            n_states,
            n_actions
        )
    ) / n_actions

    V = np.zeros(
        n_states
    )

    while True:
        # Policy Evaluation
        V = policy_evaluation(
            env,
            n_states,
            policy,
            V,
            gamma=0.99,
            theta=1e-8,
        )

        # Policy Improvement
        new_policy = policy_improvement(
            env,
            n_states,
            n_actions,
            V,
            gamma=0.99
        )

        # Check stability
        if np.array_equal(
            policy,
            new_policy
        ):
            break

        policy = new_policy

    return policy, V