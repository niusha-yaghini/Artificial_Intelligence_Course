# Markov Decision Process (MDP) and Dynamic Programming

## Overview

After studying Multi-Armed Bandits, where the agent only learns the value of actions through rewards, this section introduces a more general Reinforcement Learning framework: **Markov Decision Processes (MDPs)**.

In real-world problems, actions affect future states, and the agent must consider both immediate rewards and long-term consequences.

MDP provides a mathematical framework to model these environments and solve them using Dynamic Programming methods.

The main goal is to find an optimal policy:

\[
\pi^*
\]

that maximizes expected future rewards.

---

# 1. Markov Decision Process (MDP)

A Markov Decision Process is defined by five components:

\[
(S,A,P,R,\gamma)
\]

---

## State (S)

The set of possible situations in the environment.

For FrozenLake:

\[
S=\{0,1,...,15\}
\]

The environment contains:


4 x 4 Grid
16 possible states

Each state represents the current position of the agent.

---

## Action (A)

The set of possible actions available to the agent.

FrozenLake contains four actions:

| Action | Movement |
|-|-|
|0|Left|
|1|Down|
|2|Right|
|3|Up|

---

## Transition Probability (P)

Defines how the environment changes after an action.

\[
P(s'|s,a)
\]

For deterministic FrozenLake:


Probability = 1.0

Example:


State 0 + Action Down
→ State 4
Probability = 1

The transition model was extracted using:

```python
env.unwrapped.P

Reward Function (R)
Defines the feedback received from the environment.
FrozenLake:
Reach Goal → Reward = 1

Other moves → Reward = 0

The reward is received when transitioning into the goal state.
Discount Factor (γ)
Controls the importance of future rewards.
\[
0 \leq \gamma \leq 1
\]
In this project:
gamma = 0.99


A higher gamma makes the agent consider long-term rewards.
2. Markov Property
The Markov assumption states:
The future depends only on the current state, not the complete history.

Mathematically:
\[
P(s_{t+1}|s_t)
\]
is sufficient to describe future transitions.
This allows reinforcement learning algorithms to reason about states independently.
3. FrozenLake Environment
The environment used:
FrozenLake-v1


Grid:
S F F F
F H F H
F F F H
H F F G

Meaning:
Symbol	Description
S	Starting point
F	Frozen surface
H	Hole / Failure
G	Goal


The environment was configured as:
is_slippery=False


Therefore transitions were deterministic.
4. Random Policy Baseline
Before applying Dynamic Programming, a random agent was evaluated.
The agent selects actions randomly:
action = env.action_space.sample()


No:
- Learning
- Value estimation
- Policy improvement
is performed.
Random Agent Result
Experiment:
Episodes: 1000

Result:
Success Rate ≈ 1.5%

Analysis
The random policy rarely reaches the goal because:
- It has no knowledge of the environment.
- It cannot avoid holes.
- It does not improve over time.
This provides a baseline for evaluating learned policies.
5. Policy Iteration
Concept
Policy Iteration finds the optimal policy by repeatedly improving an existing policy.
The algorithm consists of two main steps:
Policy Evaluation

        ↓

Policy Improvement

        ↓

Repeat until stable

Policy Evaluation
Goal:
Estimate the value of the current policy.
\[
V^\pi(s)
\]
The Bellman Expectation Equation is used:
\[
V^\pi(s)=
\sum_a \pi(a|s)
\sum_{s'}
P(s'|s,a)
(R+\gamma V(s'))
\]
The algorithm calculates how valuable each state is under the current policy.
Policy Improvement
After obtaining the value function, the algorithm evaluates possible actions.
The action value function:
\[
Q(s,a)
\]
is calculated:
\[
Q(s,a)=
\sum_{s'}
P(s'|s,a)
(R+\gamma V(s'))
\]
The best action is selected:
\[
a^*=argmax Q(s,a)
\]
The policy is updated.
Policy Iteration Implementation
Implemented components:
policy_evaluation()

calculate_q_value()

policy_improvement()

policy_iteration()

The algorithm started from:
Random Policy

and repeatedly improved it.
Policy Iteration Results
Final Value Function
[[0.95099005 0.96059601 0.970299   0.96059601]
 [0.96059601 0.         0.9801     0.        ]
 [0.970299   0.9801     0.99       0.        ]
 [0.         0.99       1.         0.        ]]

Interpretation:
- States closer to the goal have higher values.
- Hole states have zero value.
- Goal state is terminal.
Learned Policy
Final policy:
0 ↓
1 →
2 ↓
3 ←

4 ↓
6 ↓

8 →
9 ↓
10 ↓

13 →
14 →

The policy avoids holes and moves toward the goal.
Evaluation
The learned agent achieved:
Reward: 1

Steps: 6

Compared to:
Random Policy:
≈1.5% success

Policy Iteration successfully learned an optimal strategy.
6. Value Iteration
Concept
Value Iteration improves the value function directly without explicitly evaluating a policy.
Instead of:
Policy Evaluation
        +
Policy Improvement

it directly applies the Bellman Optimality Equation.
Bellman Optimality Equation
\[
V(s)=
max_a
\sum_{s'}
P(s'|s,a)
(R+\gamma V(s'))
\]
For every state:
1. Calculate all possible action values.
2. Select the maximum value.
3. Update the state value.
Value Iteration Implementation
Implemented:
value_iteration()

extract_policy()

The algorithm starts with:
V = np.zeros(states)


and iteratively updates values until convergence.
Value Iteration Results
Final Value Function
The result was identical to Policy Iteration:
[[0.95099005 0.96059601 0.970299   0.96059601]
 [0.96059601 0.         0.9801     0.        ]
 [0.970299   0.9801     0.99       0.        ]
 [0.         0.99       1.         0.        ]]

Extracted Policy
The optimal policy matched Policy Iteration:
0 ↓
1 →
2 ↓
3 ←

4 ↓
6 ↓

8 →
9 ↓
10 ↓

13 →
14 →

7. Policy Iteration vs Value Iteration
Feature	Policy Iteration	Value Iteration
Main idea	Improve policy repeatedly	Optimize value directly
Steps	Evaluation + Improvement	Bellman optimal update
Uses explicit policy evaluation	Yes	No
Speed on small problems	Good	Good
Large environments	Can be expensive	Usually preferred


Final Comparison
Both algorithms successfully solved FrozenLake.
Results:
Algorithm	Reward	Steps
Random Policy	0/1 occasionally	Variable
Policy Iteration	1	6
Value Iteration	1	6


Both Dynamic Programming methods discovered the optimal navigation strategy.
Key Takeaways
- MDP provides the mathematical foundation for reinforcement learning.
- The transition model allows planning without trial-and-error.
- Value functions estimate future expected rewards.
- Policy Iteration alternates between evaluating and improving policies.
- Value Iteration directly optimizes the value function.
- Both algorithms converge to the optimal policy when the environment model is known.
This section builds the foundation for moving from model-based reinforcement learning to model-free methods such as Monte Carlo Learning, Temporal Difference Learning, and Q-Learning.
```