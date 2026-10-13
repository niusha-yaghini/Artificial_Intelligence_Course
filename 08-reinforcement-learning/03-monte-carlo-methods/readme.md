# Monte Carlo Reinforcement Learning

## Overview

After studying Markov Decision Processes and Dynamic Programming, we move from **Model-Based Reinforcement Learning** to **Model-Free Reinforcement Learning**.

In Dynamic Programming methods:

- The environment model was known.
- Transition probabilities:

\[
P(s'|s,a)
\]

were available.
- The agent could perform planning without direct experience.

However, in many real-world problems, the transition model is unknown.

Monte Carlo methods solve this problem by learning directly from experience.

The agent:

1. Interacts with the environment.
2. Generates complete episodes.
3. Calculates observed returns.
4. Estimates value functions from experience.

---

# Model-Free Reinforcement Learning

The main difference:

## Dynamic Programming

Requires:


Environment Model
P(s'|s,a)
-
Reward Function

Uses:


Bellman Equations

---

## Monte Carlo Learning

Requires only:


Experience Samples

The learning process:


Episode
↓
Rewards
↓
Return Calculation
↓
Value Estimation

---

# Blackjack Environment

The environment used:

```python
Blackjack-v1

Blackjack is suitable for Monte Carlo learning because:
- State space is discrete.
- Episodes terminate naturally.
- Rewards are received at the end of games.
State Representation
The state consists of:
\[
(player\_sum, dealer\_card, usable\_ace)
\]
Example:
(16,8,0)


means:
Player sum = 16

Dealer showing card = 8

No usable ace

Action Space
The environment has two actions:
Action	Meaning
0	Stick
1	Hit


Part 1 — Monte Carlo Prediction
Goal
Monte Carlo Prediction answers:
How good is a given policy?

The objective is estimating:
\[
V^\pi(s)
\]
which represents the expected future return from state s while following policy π.
Return Calculation
Monte Carlo methods use the return:
\[
G_t =
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2R_{t+3}
+...
\]
Instead of calculating expected future rewards using a model, the algorithm averages observed returns from actual episodes.
Example:
Episode rewards:
[0,0,1]

With:
\[
\gamma=1
\]
The returns become:
[1,1,1]

Episode Generation
A random policy was used initially:
action = env.action_space.sample()


Each episode was stored as:
(state, action, reward)


Example:
[((14,4,0),1,0.0),((18,4,0),0,1.0)]


Meaning:
State (14,4,0)

Action Hit

Reward 0


State (18,4,0)

Action Stick

Reward +1

First-Visit Monte Carlo
Idea
For each state:
Only the first occurrence inside an episode is used.
Example:
A → B → C → B → Goal

For state B:
Only:
First B

is included.
Implementation
The algorithm:
1. Generates many episodes.
2. Calculates returns.
3. Stores returns for each state.
4. Computes average return.
Formula:
\[
V(s)=Average(G_t)
\]
Result
After 10,000 episodes:
Observed States:

280

The estimated value:
Example:
State:
(20,8,0)

Result:
V(s) = -0.03125

Interpretation:
Under the random policy, this state had a slightly negative expected return.
Although the player has a strong hand, the random policy may still choose poor actions.
Value Function Visualization
The estimated values were visualized using heatmaps.
Two cases were analyzed:
1. No usable ace
2. Usable ace
Observations
The results showed:
- Strong player hands generally had higher values.
- Strong dealer cards reduced expected returns.
- Weak dealer cards increased player advantage.
Because the policy was random, the heatmaps contained noise.
The values represented:
Expected return under random behavior

not an optimal Blackjack strategy.
Every-Visit Monte Carlo
Idea
Every-Visit Monte Carlo uses every occurrence of a state.
Example:
A → B → C → B → Goal

Both occurrences of B are used.
Comparison
First-Visit:
- Uses fewer samples.
- More independent samples.
- Higher variance.
Every-Visit:
- Uses more experience.
- Usually smoother estimation.
- Lower variance.
Result Comparison
For:
State = (20,8,0)


Results:
Method	Value
First-Visit MC	-0.0165
Every-Visit MC	0.1188


The difference comes from the different sampling strategies.
Part 2 — Monte Carlo Control
Goal
Prediction estimates:
\[
V(s)
\]
but control requires learning the best actions.
Therefore we estimate:
\[
Q(s,a)
\]
Action Value Function
\[
Q(s,a)=E[G_t|S_t=s,A_t=a]
\]
It answers:
How good is taking action a in state s?

Example:
State:
(20,9,0)

Actions:
Stick

Hit

The agent learns:
Q(state,Stick)

Q(state,Hit)

and selects:
\[
argmax_a Q(s,a)
\]
Q-table Initialization
Instead of:
V[state]


we use:
Q[state][action]


Implementation:
Q = defaultdict(    lambda: np.zeros(        env.action_space.n    ))


Each state contains values for:
Stick
Hit

Exploration Strategy — ε-Greedy
To balance exploration and exploitation:
With probability:
\[
\epsilon
\]
choose a random action.
Otherwise:
choose the best known action.
Implementation:
if random < epsilon:    random actionelse:    argmax(Q[state])


Monte Carlo Control Algorithm
The training loop:
Initialize Q-table

↓

Generate episode using ε-greedy policy

↓

Calculate returns

↓

Update Q(s,a)

↓

Improve policy

↓

Repeat

Learned Policy Examples
After training:
Strong Hand vs Dealer 9
State:
(20,9,0)

Q-values:
Stick: 0.799

Hit: -0.812

Selected:
Stick

The agent correctly learned that hitting on 20 is dangerous.
Weak Hand vs Dealer 10
State:
(13,10,0)

Q-values:
Stick: -0.584

Hit: -0.458

Selected:
Hit

The agent learned that staying on 13 against a strong dealer is usually worse.
Low Hand vs Dealer 2
State:
(12,2,0)

Q-values:
Stick: -0.562

Hit: -0.226

Selected:
Hit

Policy Evaluation
The learned policy was compared against a random baseline.
Evaluation metric:
Average reward over many games.
Results:
Agent	Average Reward
Random Policy	-0.3964
Monte Carlo Control	-0.0556


Analysis
The Monte Carlo Control agent significantly outperformed the random policy.
The improvement demonstrates that the agent successfully learned:
- When to Hit.
- When to Stick.
- How different states affect future returns.
The final reward remained negative because:
- Blackjack is stochastic.
- Dealer outcomes are random.
- More training episodes can further improve convergence.
Key Takeaways
Monte Carlo Prediction
Learn:
\[
V(s)
\]
from complete episodes.
Monte Carlo Control
Learn:
\[
Q(s,a)
\]
and improve policy through experience.
Main Concepts Learned
- Model-Free RL
- Episode-based learning
- Return estimation
- First-Visit Monte Carlo
- Every-Visit Monte Carlo
- Q-value estimation
- ε-Greedy exploration
- Policy improvement
Final Pipeline
The progression from previous sections:
Multi-Armed Bandit

↓

MDP + Dynamic Programming

↓

Monte Carlo Prediction

↓

Monte Carlo Control

↓

Temporal Difference Learning

Monte Carlo methods provide the bridge between planning-based RL and modern value-based algorithms such as SARSA and Q-Learning.
```