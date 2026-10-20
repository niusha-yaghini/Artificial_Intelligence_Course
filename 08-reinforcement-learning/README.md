# Reinforcement Learning Portfolio

This repository contains a collection of hands-on Reinforcement Learning (RL) projects, organized as a learning path from foundational decision-making algorithms to deep reinforcement learning.

Each section focuses on a specific family of methods and includes Jupyter notebooks, supporting Python modules where applicable, and a section-level README. The goal is to understand core concepts, implement algorithms, inspect learning behavior, and document experimental results.

## Project Roadmap

| Folder | Topic | Main concepts |
|---|---|---|
| `01-multi-armed-bandit` | Multi-Armed Bandits | Exploration vs. exploitation, epsilon-greedy, UCB, Thompson sampling |
| `02-mdp-dynamic-programming` | MDPs and Dynamic Programming | Markov Decision Processes, policy iteration, value iteration |
| `03-monte-carlo-methods` | Monte Carlo Methods | Episode-based prediction and control |
| `04-temporal-difference-learning` | Temporal-Difference Learning | TD prediction, SARSA, Q-Learning |
| `05-deep-q-network` | Deep Q-Network | Neural Q-value approximation, replay buffer, target network, library comparison |
| `06-policy-gradient` | Policy Gradient | REINFORCE, value-function baseline, advantage estimates |

## Folder Guide

### 01 — Multi-Armed Bandit

**Purpose:** Explore how an agent balances exploration and exploitation when choosing among actions with uncertain rewards.

Notebooks:
- `00_epsilon_greedy_baseline.ipynb` — baseline epsilon-greedy strategy.
- `01_epsilon_decay_greedy.ipynb` — epsilon-greedy with a decaying exploration rate.
- `02_ucb_algorithm.ipynb` — Upper Confidence Bound (UCB).
- `03_thompson_sampling.ipynb` — Thompson sampling.
- `04_compare_strategies.ipynb` — comparison of the implemented strategies.

Supporting code:
- `src/bandit.py` — shared bandit implementation.

### 02 — MDP and Dynamic Programming

**Purpose:** Introduce Markov Decision Processes and planning algorithms that use a model of the environment.

Notebooks:
- `01_frozenlake_mdp.ipynb` — explore the FrozenLake environment as an MDP.
- `02_policy_iteration.ipynb` — solve for a policy using policy iteration.
- `03_value_iteration.ipynb` — derive a policy using value iteration.

Supporting code:
- `src/policy_iteration.py`
- `src/value_iteration.py`

### 03 — Monte Carlo Methods

**Purpose:** Learn from complete episodes and use sampled returns to estimate values or improve a policy.

Notebooks:
- `01_monte_carlo_prediction.ipynb` — Monte Carlo prediction.
- `02_monte_carlo_control.ipynb` — Monte Carlo control.

Supporting code:
- `src/mont_calro.py` — supporting Monte Carlo implementation (current filename).

### 04 — Temporal-Difference Learning

**Purpose:** Study methods that update value estimates using observed rewards and estimates of subsequent states, without waiting for the full return of an episode.

Notebooks:
- `01_td_prediction.ipynb` — TD prediction.
- `02_sarsa.ipynb` — on-policy SARSA.
- `03_q_learning.ipynb` — off-policy Q-Learning.

Supporting code:
- `src/td_prediction.py`

The experiments include learning-curve analysis and evaluation of learned behavior in a grid-based environment.

### 05 — Deep Q-Network

**Purpose:** Extend value-based learning to a neural network that approximates action values in a state space with continuous observations.

Notebooks:
- `01_dqn_cartpole.ipynb` — custom DQN implementation on CartPole.
- `02_dqn_library_comparison.ipynb` — comparison with a DQN implementation from Stable-Baselines3.

Saved artifacts in `results/`:
- `custom_dqn_arrays.npz` — recorded custom-DQN arrays.
- `custom_dqn_best.pth` — saved custom-DQN model checkpoint.
- `custom_dqn_metrics.json` — recorded custom-DQN metrics.
- `dqn_mean_reward_comparison.png` — reward comparison plot.
- `library_dqn_cartpole.zip` — saved library-DQN model.
- `library_dqn_eval.npz` — library-DQN evaluation results.

This section demonstrates a custom implementation and a library-based comparison. Results depend on training settings, random seeds, and evaluation procedure; consult the section README for the recorded experiment details.

### 06 — Policy Gradient

**Purpose:** Learn a policy directly rather than choosing actions only through a learned Q-table or Q-network.

Notebook:
- `01_reinforce.ipynb` — policy-gradient experiments using REINFORCE and REINFORCE with a learned value-function baseline.

Saved artifacts in `results/`:
- `reinforce_baseline_cartpole.pth` — saved policy and value network checkpoint.
- `reinforce_baseline_results.npz` — saved training and evaluation arrays.

The recorded baseline experiment reached a mean reward of 500 over 100 greedy evaluation episodes and 100 stochastic evaluation episodes on `CartPole-v1`. This is the result of the saved experiment, not a guarantee for every run or environment configuration.

## Learning Progression

The folders are ordered to build concepts gradually:

1. **Bandits:** learn exploration and exploitation when there is no state-transition model to plan over.
2. **MDPs and Dynamic Programming:** represent sequential decision problems and solve small modeled environments with planning.
3. **Monte Carlo:** estimate values from complete sampled episodes.
4. **Temporal Difference:** learn from immediate rewards and bootstrapped estimates; compare SARSA and Q-Learning.
5. **Deep Q-Network:** approximate action values with a neural network and use replay and target networks for training.
6. **Policy Gradient:** optimize a stochastic policy directly and use a value baseline to form advantage estimates.

Together, these sections cover several central RL ideas: planning, learning from sampled experience, bootstrapping, value-based deep learning, and direct policy optimization.

## Tools and Libraries

The projects use Python notebooks and supporting modules. Depending on the section, the main libraries include:

- **NumPy** for numerical operations and saved arrays.
- **Matplotlib** for learning curves and result visualizations.
- **Gymnasium** for reinforcement-learning environments.
- **PyTorch** for neural-network implementations.
- **Stable-Baselines3** for the library-based DQN comparison in Section 05.

Install the dependencies needed by a particular notebook in your Python environment. For example:

```bash
pip install numpy matplotlib gymnasium torch stable-baselines3 jupyter
```

Package requirements can vary by notebook and operating system. If a notebook uses additional packages, install those as indicated by its section README or import errors.

## How to Use This Repository

1. Start with the numbered folders and proceed in order, or open a section that matches the concept you want to study.
2. Read that folder's `readme.md` before running its notebooks.
3. Run notebook cells in order so that classes, functions, and environment variables are defined before use.
4. Inspect the learning curves and evaluation results instead of relying only on a single episode.
5. For the DQN and Policy Gradient sections, review the saved files in `results/` alongside the section README.
6. Treat recorded metrics as results from specific experiments; rerunning training may produce different outcomes.

## Repository Notes

- `__pycache__/` directories contain Python-generated bytecode and are not source code. They generally do not need to be tracked in version control.
- Model checkpoints and `.npz` files are generated artifacts. Keep them when they are useful for evaluation or reproducibility, but avoid committing large files that are not needed.
- Each section README is the source for detailed algorithm explanations, implementation notes, and experiment-specific results.
- The directory names and notebook names are preserved as they currently appear in the project structure.

## Overall Goal

This portfolio is a practical, incremental study of Reinforcement Learning. It moves from simple action-selection strategies to model-based planning, sample-based learning, deep value estimation, and policy optimization. The emphasis is on understanding how each method works, implementing or using it in a controlled environment, and interpreting its behavior through documented experiments.