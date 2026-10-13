import numpy as np
import pandas as pd
from collections import defaultdict


# ----------------------------------
# Monte Carlo Prediction 
# ----------------------------------

def random_policy(env, state):
    return env.action_space.sample()

def generate_episode(env):
    episode = []
    state, info = env.reset()

    while True:
        action = random_policy(
            env,
            state
        )
        next_state, reward, terminated, truncated, info = env.step(
            action
        )
        episode.append(
            (
                state,
                action,
                reward
            )
        )
        state = next_state

        if terminated or truncated:
            break

    return episode


def compute_returns(
    episode,
    gamma=1.0
):
    returns = []
    G = 0

    for state, action, reward in reversed(episode):
        G = reward + gamma * G
        returns.append(
            (
                state,
                G
            )
        )

    return list(
        reversed(returns)
    )
    
    
def first_visit_mc(
    env,
    episodes,
    gamma=1.0
):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = {}

    for _ in range(episodes):
        episode = generate_episode(env)
        visited = set()

        for state, G in compute_returns(
            episode,
            gamma
        ):
            if state not in visited:
                returns_sum[state] += G
                returns_count[state] += 1
                visited.add(state)

    for state in returns_sum:
        V[state] = (
            returns_sum[state]
            /
            returns_count[state]
        )

    return V


def every_visit_mc(
    env,
    episodes,
    gamma=1.0
):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = {}

    for _ in range(episodes):
        episode = generate_episode(env)

        for state, G in compute_returns(
            episode,
            gamma
        ):
            returns_sum[state] += G
            returns_count[state] += 1

    for state in returns_sum:
        V[state] = (
            returns_sum[state]
            /
            returns_count[state]
        )

    return V
    
    
# ----------------------------------
# Monte Carlo Control 
# ----------------------------------

def epsilon_greedy(
    env,
    state,
    Q,
    epsilon=0.1
):
    if np.random.random() < epsilon:
        return env.action_space.sample()

    return np.argmax(
        Q[state]
    )
    
    
def control_generate_episode(
    env,
    Q,
    epsilon=0.1
):
    episode = []
    state, info = env.reset()

    while True:
        action = epsilon_greedy(
            env,
            state,
            Q,
            epsilon
        )

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        episode.append(
            (
                state,
                action,
                reward
            )
        )

        state = next_state

        if terminated or truncated:
            break

    return episode


def monte_carlo_control(
    env,
    episodes,
    epsilon=0.1,
    gamma=1.0
):
    Q = defaultdict(
        lambda: np.zeros(
            env.action_space.n
        )
    )

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for _ in range(episodes):
        episode = control_generate_episode(
            env,
            Q,
            epsilon
        )

        G = 0
        visited = set()

        for state, action, reward in reversed(episode):
            G = reward + gamma * G

            state_action = (
                state,
                action
            )

            if state_action not in visited:
                returns_sum[state_action] += G
                returns_count[state_action] += 1

                Q[state][action] = (
                    returns_sum[state_action]
                    /
                    returns_count[state_action]
                )

                visited.add(
                    state_action
                )

    return Q