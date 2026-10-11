import numpy as np

def value_iteration(
    env,
    n_states,
    n_actions,
    gamma=0.99,
    theta=1e-8
):
    V = np.zeros(
        n_states
    )

    while True:
        delta = 0

        for state in range(n_states):
            q_values = []

            for action in range(n_actions):
                q = 0

                for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                    q += prob * (
                        reward
                        +
                        gamma * V[next_state]
                    )

                q_values.append(q)

            best_value = max(
                q_values
            )

            delta = max(
                delta,
                abs(
                    V[state]-best_value
                )
            )

            V[state] = best_value

        if delta < theta:
            break

    return V


def extract_policy(
    env,
    n_states,
    n_actions,
    V,
    gamma,
):
    policy = np.zeros(
        (
            n_states,
            n_actions
        )
    )

    for state in range(n_states):
        q_values = []

        for action in range(n_actions):
            q = 0
            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                q += prob * (
                    reward
                    +
                    gamma * V[next_state]
                )
            q_values.append(q)

        best_action = np.argmax(
            q_values
        )

        policy[state][best_action] = 1

    return policy