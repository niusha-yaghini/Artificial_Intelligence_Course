# Multi-Armed Bandit Strategies

## Overview

Multi-Armed Bandit is one of the simplest reinforcement learning problems and provides the foundation for understanding the exploration-exploitation trade-off.

The problem consists of multiple actions (arms), where each action provides a reward from an unknown probability distribution.

The agent must learn which action provides the highest expected reward while balancing:

- **Exploration:** Trying new actions to gather information.
- **Exploitation:** Selecting the best-known action based on current knowledge.

Unlike supervised learning, there is no labeled dataset. The agent learns entirely through interaction with the environment and reward feedback.

---

# 1. Multi-Armed Bandit Environment

A bandit environment contains:

- A set of available actions (arms)
- An unknown reward distribution for each arm
- A reward returned after selecting an action

For a Gaussian Bandit:

\[
Reward \sim N(\mu_a, \sigma^2)
\]

where:

- \( \mu_a \) is the unknown true value of arm `a`
- The agent must estimate these values through experience

The goal is to maximize total reward and minimize regret.

---

# 2. Evaluation Metrics

## Average Reward

Average reward shows how quickly the agent improves:

\[
AverageReward =
\frac{\sum_{i=1}^{t}R_i}{t}
\]

A better algorithm should converge faster to the optimal reward.

---

## Regret

Regret measures the difference between the optimal action and the selected action.

Instant regret:

\[
Regret_t =
\mu^*-\mu_{a_t}
\]

where:

- \( \mu^* \) is the reward of the best arm
- \( \mu_{a_t} \) is the selected arm reward

Cumulative regret:

\[
CumulativeRegret =
\sum_{t=1}^{T}Regret_t
\]

A better algorithm achieves lower regret.

---

# 3. Epsilon Greedy Baseline

## Idea

The simplest exploration strategy.

The agent chooses:

- Best known action with probability \(1-\epsilon\)
- Random action with probability \(\epsilon\)

Example:


epsilon = 0.1
90% → exploit best action
10% → explore random action

---

## Implementation

The agent maintains:

- Estimated value of each arm
- Number of times each arm was selected

The estimated value is updated using incremental mean:

\[
Q(a)=Q(a)+\frac{1}{N}(R-Q(a))
\]

---

## Limitation

Fixed epsilon creates unnecessary exploration.

Even after learning the best action, the agent still randomly selects other arms.

---

# 4. Epsilon Decay Greedy

## Motivation

Fixed epsilon does not adapt during learning.

At the beginning:

- The agent knows nothing.
- Exploration is important.

Later:

- The agent has enough information.
- Exploitation is more valuable.

Therefore epsilon is gradually reduced.

---

## Formula

\[
\epsilon_t=\epsilon_0 \times decay^t
\]


Example:


Initial epsilon = 1.0
↓
Explore heavily
Later:
epsilon ≈ 0.01
↓
Mostly exploit

---

## Implementation

After every step:

```python
epsilon = max(
    min_epsilon,
    epsilon * decay
)

The agent transitions from exploration to exploitation automatically.
5. Upper Confidence Bound (UCB)
Idea
UCB replaces random exploration with uncertainty-based exploration.
Instead of choosing only the highest estimated value:
\[
Q(a)
\]
it chooses:
\[
UCB(a)=Q(a)+ConfidenceBonus
\]
The confidence term encourages selecting actions with limited information.
Intuition
If an arm:
- Has high reward estimate → exploit
- Has few observations → explore
The algorithm balances both automatically.
Advantages
- No epsilon parameter
- Exploration is directed
- Usually achieves lower regret than random exploration methods
6. Thompson Sampling
Idea
Thompson Sampling is a Bayesian approach to exploration.
Instead of maintaining only one value for each arm, the agent maintains a probability distribution representing its belief.
Example:
Instead of:
Arm 1 = 0.7

we maintain:
Arm 1 ~ Beta(alpha, beta)

Bayesian Update
Initially:
\[
Beta(1,1)
\]
After observing rewards:
Success:
\[
\alpha += 1
\]
Failure:
\[
\beta += 1
\]
The posterior distribution becomes more accurate over time.
Action Selection
At every step:
1. Sample from each arm posterior.
2. Select the arm with the highest sampled value.
This naturally balances exploration and exploitation.
Experiments
Gaussian Bandit Comparison
The following algorithms were compared:
- Epsilon Greedy
- Epsilon Decay Greedy
- UCB
Experiment settings:
Number of arms: 10

Training steps: 5000

Learning Curve Results
The final average rewards:
Algorithm	Final Average Reward
Epsilon Greedy	1.470
Epsilon Decay	1.587
UCB	1.569


Analysis
Epsilon Greedy learned quickly but stopped improving because exploration remained fixed.
Epsilon Decay achieved the best reward because it explored early and exploited later.
UCB performed similarly by using uncertainty-based exploration instead of random exploration.
Cumulative Regret Results
Algorithm	Final Cumulative Regret
Epsilon Greedy	620
Epsilon Decay	254
UCB	291


Analysis
Epsilon Greedy produced the highest regret because random exploration continued throughout training.
Epsilon Decay achieved the lowest regret because exploration was reduced as the agent became more confident.
UCB also achieved low regret by selecting actions according to both value and uncertainty.
Thompson Sampling Experiment
For Thompson Sampling, a Bernoulli Bandit environment was used.
Each arm returned:
Success = 1
Failure = 0

The agent learned the success probability of each arm using Beta distributions.
Posterior Estimation Result
The final posterior means showed that the agent correctly learned the best arm.
Example:
True best arm:
2

Agent best arm:
2

The posterior parameters showed that the optimal arm received significantly more observations:
Alpha:
[4, 2, 3946]

Beta:
[12, 8, 1034]

The third arm accumulated many more observations because Thompson Sampling identified it as the most promising action.
Thompson Sampling Analysis
The learning curve showed rapid convergence toward the optimal reward.
The cumulative regret increased only during the initial exploration phase and became almost constant after identifying the best arm.
This demonstrates the efficiency of Bayesian exploration.
Final Comparison
Algorithm	Exploration Strategy	Main Advantage
Epsilon Greedy	Random exploration	Simple baseline
Epsilon Decay	Adaptive random exploration	Better long-term exploitation
UCB	Uncertainty-based exploration	No epsilon tuning
Thompson Sampling	Bayesian posterior sampling	Natural exploration-exploitation balance


Key Takeaways
- Exploration and exploitation are the central challenges in reinforcement learning.
- Fixed exploration rates can waste interactions after learning.
- Adaptive strategies improve learning efficiency.
- UCB uses uncertainty to guide exploration.
- Thompson Sampling applies Bayesian reasoning to decision making.
- Multi-Armed Bandit provides the foundation for more advanced RL algorithms such as MDPs, Q-Learning, and Policy Gradient methods.