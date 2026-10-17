Temporal Difference Learning — From Prediction to Control

1. Overview

This project explores Temporal Difference (TD) Learning, a fundamental family of model-free reinforcement learning algorithms.

The goal is to understand how an agent can learn state values and action values directly from experience, without requiring a complete model of the environment.

The project progresses from value prediction to control:

TD(0) Prediction — estimate the value of each state under a policy.

SARSA — learn state-action values using an on-policy update.

Q-Learning — learn state-action values using an off-policy update.

The prediction experiments also compare TD(0) with Monte Carlo prediction, while the control experiments compare SARSA and Q-Learning in the CliffWalking-v1 environment.

Learning objectives

Understand the difference between prediction and control.

Understand the TD error and bootstrapping.

Compare Monte Carlo and TD(0) learning.

Implement SARSA and Q-Learning from scratch using NumPy.

Understand on-policy and off-policy learning.

Evaluate policies using cumulative reward, episode length, success rate, and cliff falls.

Visualize learning curves and learned trajectories.

Understand why training performance and greedy evaluation performance can differ.

2. Repository Structure

04_temporal_difference_learning/
│
├── 01_td_prediction.ipynb
├── 02_sarsa.ipynb
├── 03_q_learning.ipynb
└── README.md

Notebook

Purpose

01_td_prediction.ipynb

TD(0) state-value prediction and comparison with Monte Carlo

02_sarsa.ipynb

On-policy TD control using SARSA

03_q_learning.ipynb

Off-policy TD control using Q-Learning

The notebooks progressively introduce the theory, implement the algorithms, run experiments, and analyze the resulting value estimates and policies.

Part I — Theoretical Foundations

3. Reinforcement Learning Fundamentals

Reinforcement Learning (RL) studies how an agent learns to make decisions by interacting with an environment.

At each time step, the agent:

Observes the current state (S_t).

Selects an action (A_t).

Receives a reward (R_{t+1}).

Transitions to a new state (S_{t+1}).

The interaction can be represented as:

[
S_t \rightarrow A_t \rightarrow R_{t+1},S_{t+1}
]

The objective is to learn a policy that maximizes expected cumulative reward.

Important terminology

Term

Meaning

State (s)

The current situation of the agent

Action (a)

A decision available to the agent

Reward (r)

Immediate feedback from the environment

Policy (\pi)

A rule for selecting actions

Value function (V(s))

Expected return from a state under a policy

Action-value function (Q(s,a))

Expected return from taking an action in a state and subsequently following a policy

Return (G_t)

Discounted sum of future rewards

Learning rate (\alpha)

Controls the size of each update

Discount factor (\gamma)

Controls the importance of future rewards

The discounted return is:

[
G_t=\sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}
]

where (T) is the terminal time for a finite episode.

4. Prediction vs. Control

Two fundamental problems appear throughout reinforcement learning.

4.1 Prediction

Prediction estimates how good states are under a given policy.

The state-value function is:

[
V^\pi(s)=
\mathbb{E}_\pi[G_t\mid S_t=s]
]

The question is:

If the agent follows this policy, how much reward can it expect from a given state?

In prediction, the policy is fixed. The objective is to estimate its value rather than directly optimize its actions.

4.2 Control

Control attempts to find a policy that maximizes expected return.

The action-value function is:

[
Q^\pi(s,a)=
\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]
]

The question is:

Which action should the agent choose in each state to obtain the highest expected return?

SARSA and Q-Learning solve this control problem by learning action values and using them to improve action selection.

Key distinction

Prediction

Control

Estimates (V(s)) or (Q^\pi(s,a)) for a policy

Learns action values to improve a policy

Evaluates behavior

Improves behavior

TD(0) is used in our state-value experiment

SARSA and Q-Learning are used in our control experiments

5. Monte Carlo vs. Temporal Difference Learning

Monte Carlo (MC) and Temporal Difference (TD) methods are both model-free learning approaches.

Their main difference is how they construct the target used to update value estimates.

5.1 Monte Carlo Prediction

Monte Carlo prediction waits until an episode finishes and then uses the observed return to update the value of a visited state.

A basic update is:

[
V(S_t)\leftarrow V(S_t)+
\alpha[G_t-V(S_t)]
]

Here, (G_t) is the actual return observed after completing the episode.

Advantages

Uses the complete observed return.

Does not bootstrap from an estimated future value.

Is conceptually straightforward.

Limitations

Must wait until the episode ends.

Return estimates can have high variance.

Learning can be slow when episodes are long.

5.2 Temporal Difference Prediction

TD learning updates a value estimate using an observed reward and an estimate of the next state's value.

The TD(0) update is:

[
V(S_t)\leftarrow V(S_t)+
\alpha[
R_{t+1}+\gamma V(S_{t+1})-V(S_t)
]
]

The update can be performed after each transition, without waiting for the episode to finish.

Advantages

Learns incrementally.

Can update values after every transition.

Bootstrapping can improve learning efficiency.

Limitations

Updates depend on current value estimates.

Results can fluctuate as estimates change.

TD targets can be biased because they use estimated values.

5.3 What is bootstrapping?

Bootstrapping means using an existing estimate to improve another estimate.

For TD(0), the target is:

[
\text{TD Target}=R_{t+1}+\gamma V(S_{t+1})
]

The TD error is:

[
\delta_t=
R_{t+1}+\gamma V(S_{t+1})-V(S_t)
]

The TD error measures the difference between the updated target and the current estimate.

Positive TD error: the target is greater than the current estimate.

Negative TD error: the target is smaller than the current estimate.

Zero TD error: no change is made by that update.

For a terminal transition, the next-state value is treated as zero, so the target becomes the immediate reward.

5.4 Bias and variance

Monte Carlo uses a complete sampled return, whereas TD(0) bootstraps from a current estimate.

In general:

MC does not bootstrap and can have high-variance targets.

TD(0) bootstraps and can introduce bias through its estimated next-state value.

Neither method is universally better in every environment or at every stage of training.

Their practical performance depends on the environment, learning rate, policy, sampling process, and amount of experience.

Part II — TD(0) Prediction

6. Notebook 01: TD(0) Prediction

The first notebook implements state-value prediction and compares TD(0) with Monte Carlo.

The experiment uses a small random-walk-style environment with five states:

[
S={0,1,2,3,4}
]

States 0 and 4 are terminal states. States 1, 2, and 3 are nonterminal states.

The reference values shown in the experiment are approximately:

[
V_{\text{true}}=[0,0.25,0.50,0.75,0]
]

The terminal-state values are represented as zero. The intermediate reference values provide a target for evaluating prediction accuracy.

6.1 TD(0) implementation

The essential update is:

td_target = reward + gamma * V[next_state]

td_error = td_target - V[state]

V[state] += alpha * td_error

For a terminal transition, the next-state value must not contribute to the target:

if terminated:
    td_target = reward
else:
    td_target = reward + gamma * V[next_state]

td_error = td_target - V[state]

V[state] += alpha * td_error

The exact reward and transition logic depends on the environment implementation.

6.2 Monte Carlo implementation

Monte Carlo records an episode and calculates the return from each visited time step.

The return is calculated backwards:

[
G_t=R_{t+1}+\gamma G_{t+1}
]

The value estimate is then updated using the observed return:

V[state] += alpha * (return_value - V[state])

The essential difference is that MC uses a complete return, while TD(0) uses the next state's current value estimate.

6.3 Learning-curve comparison

The notebook plots RMSE against the number of training episodes.

The Root Mean Squared Error is:

[
RMSE=
\sqrt{
\frac{1}{|\mathcal{S}|}
\sum_{s\in\mathcal{S}}
\left(V(s)-V_{\text{true}}(s)\right)^2
}
]

The terminal states and the states included in the metric should be treated consistently when calculating the RMSE.

A lower RMSE indicates that the estimated state values are closer to the reference values.

6.4 Observed value estimates

One recorded estimate from the experiment was:

[0.00000000,
 0.25607328,
 0.50149490,
 0.68770657,
 0.00000000]

Compared with the reference intermediate values of approximately 0.25, 0.50, and 0.75:

State 1 was estimated at approximately 0.256.

State 2 was estimated at approximately 0.501.

State 3 was estimated at approximately 0.688.

The first two estimates are close to their reference values. The third still has a noticeable difference.

This is an observed value estimate, not a claim that every run converges to exactly these numbers.

6.5 Observed RMSE learning curve

The recorded comparison showed the following qualitative behavior:

TD(0) started with a lower RMSE than Monte Carlo in the displayed experiment.

TD(0) subsequently fluctuated, with its RMSE varying noticeably between episodes.

Monte Carlo began with a higher RMSE but decreased substantially as more episodes were observed.

By the end of the plotted experiment, the Monte Carlo curve was lower and smoother than the TD(0) curve.

The experiment illustrates an important point: an algorithm that starts with lower error does not necessarily finish with lower error.

The plotted curves support a comparison for this particular experiment. They do not establish that Monte Carlo always outperforms TD(0).

6.6 Interpretation

The TD(0) experiment demonstrates that value learning can happen after individual transitions, while Monte Carlo relies on complete returns.

The comparison also demonstrates why learning curves are useful: a single final value estimate does not show how quickly or consistently an algorithm learns.

Part III — SARSA Control

7. Notebook 02: SARSA

SARSA extends TD learning from state-value prediction to action-value learning.

Instead of learning only (V(s)), SARSA learns:

[
Q(s,a)
]

This allows the agent to compare actions available in the same state and improve its policy.

7.1 The SARSA update

The update is:

[
\begin{aligned}
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[
&R_{t+1}\
&+\gamma Q(S_{t+1},A_{t+1})\
&-Q(S_t,A_t)]
\end{aligned}
]

The SARSA target is:

[
R_{t+1}+\gamma Q(S_{t+1},A_{t+1})
]

The five elements are:

S — current state

A — current action

R — reward

S′ — next state

A′ — next action

SARSA is an on-policy algorithm because its target uses the next action actually selected by the current behavior policy.

7.2 Epsilon-greedy action selection

During training, the agent balances exploration and exploitation.

With probability (\epsilon), it chooses a random action. Otherwise, it chooses an action with the maximum Q-value.

def epsilon_greedy(state, Q, epsilon=0.1):

    if np.random.random() < epsilon:
        return env.action_space.sample()

    best_actions = np.flatnonzero(
        Q[state] == np.max(Q[state])
    )

    return int(np.random.choice(best_actions))

Random selection among tied best actions prevents the agent from always choosing the first action when all Q-values are equal.

The action meanings in CliffWalking-v1 are:

Action

Movement

0

Up

1

Right

2

Down

3

Left

7.3 Q-table initialization

The environment has 48 discrete states and four discrete actions.

The Q-table is initialized as:

Q = np.zeros((
    env.observation_space.n,
    env.action_space.n
))

Its shape is:

(48, 4)

Each row represents a state, and each column represents an action.

All values initially equal zero because no learning has occurred yet.

7.4 SARSA training loop

The essential logic is:

state, _ = env.reset()

action = epsilon_greedy(state, Q, epsilon)

while True:

    next_state, reward, terminated, truncated, _ = (
        env.step(action)
    )

    if terminated:
        td_target = reward

        Q[state, action] += alpha * (
            td_target - Q[state, action]
        )

        break

    next_action = epsilon_greedy(
        next_state, Q, epsilon
    )

    td_target = (
        reward
        + gamma * Q[next_state, next_action]
    )

    Q[state, action] += alpha * (
        td_target - Q[state, action]
    )

    if truncated:
        break

    state = next_state
    action = next_action

This is the core algorithm used in the notebook. The surrounding training function resets the environment for each episode and records total reward and episode length.

7.5 Why SARSA differs from Q-Learning

SARSA uses the action selected by its behavior policy in the next state.

If the next action is exploratory, the target includes the Q-value of that exploratory action.

This means the learned values reflect the behavior of the policy being used during training.

8. SARSA Environment: CliffWalking-v1

The control experiments use Gymnasium's CliffWalking-v1 environment.

The grid contains four rows and twelve columns, giving 48 states.

The state index is converted to grid coordinates using:

row, col = divmod(state, 12)

The main locations are:

Location

State

Coordinates

Start

36

(3, 0)

Goal

47

(3, 11)

Cliff

37–46

Bottom row, columns 1–10

Reward structure

Every ordinary move gives a reward of -1.

Entering the cliff gives a reward of -100.

After a cliff transition, the agent returns to the start state.

Reaching the goal terminates the episode.

Falling off the cliff does not itself terminate the episode in the standard environment.

This distinction matters because an agent can fall multiple times within one episode.

Shortest safe path

The shortest safe route has 13 moves:

36 → 24 → 25 → 26 → 27 → 28 → 29
   → 30 → 31 → 32 → 33 → 34 → 35 → 47

The route first moves up from the start, then travels right across the row above the cliff, and finally moves down into the goal.

Since each of the 13 moves gives -1 and no cliff penalty occurs, the total reward is:

[
G=-13
]

9. SARSA Training Configuration

The recorded SARSA experiment used:

Parameter

Value

Meaning

Episodes

500

Number of training episodes

Learning rate (\alpha)

0.5

Size of each update

Discount factor (\gamma)

1.0

Future rewards are not discounted

Exploration rate (\epsilon)

0.1

Probability of random action selection

9.1 Training results

Metric

Result

Mean reward, first 50 episodes

-111.52

Mean reward, last 50 episodes

-22.90

Mean steps, last 50 episodes

20.92

The mean reward improved by approximately 88.62 points between the first and last 50-episode windows.

The mean episode length also decreased substantially during training.

The learning curves showed that:

Early episodes could have extremely negative rewards and very long trajectories.

The moving average of reward improved over time.

Episode lengths decreased and became more concentrated around shorter trajectories.

Occasional large negative rewards persisted during training.

These observations are consistent with an agent learning to reach the goal more effectively while still exploring.

9.2 SARSA greedy evaluation

After training, the learned Q-table was evaluated over 1,000 episodes without epsilon-based exploration.

The evaluation selected a greedy action from the learned Q-values and did not update the Q-table.

Observed results:

Metric

SARSA result

Evaluation episodes

1,000

Mean reward

-17.0

Mean steps

17.0

Mean cliff falls per episode

0.0

Success rate

100%

Timeout episodes

0

9.3 SARSA learned path

The inspected trajectory was:

[36, 24, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 35, 47]

The corresponding route:

Moves upward from State 36 to State 24.

Continues upward through State 12 to State 0.

Travels right along the top row to State 11.

Moves down through States 23 and 35.

Reaches the goal at State 47.

The path is safe and takes 17 moves.

Its total reward is -17, matching the number of moves and the absence of cliff penalties.

9.4 SARSA interpretation

SARSA learned a reliable policy that reached the goal in every evaluation episode without falling off the cliff.

However, the inspected path used four more moves than the shortest safe route.

This result demonstrates that a policy can be reliable and safe without being optimal in path length.

The result is also consistent with SARSA's on-policy update: its learning target reflects the action selected by the current epsilon-greedy policy.

Part IV — Q-Learning Control

10. Notebook 03: Q-Learning

Q-Learning is another TD control algorithm. Like SARSA, it learns the action-value function (Q(s,a)), but it uses a different target.

10.1 The Q-Learning update

The update is:

[
\begin{aligned}
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[
&R_{t+1}\
&+\gamma\max_a Q(S_{t+1},a)\
&-Q(S_t,A_t)]
\end{aligned}
]

The target is:

[
R_{t+1}+\gamma\max_a Q(S_{t+1},a)
]

The next-state contribution uses the maximum Q-value across available actions, rather than the Q-value of the next action actually selected.

10.2 Why Q-Learning is off-policy

Q-Learning is off-policy because it can learn action values for a greedy target policy while using an exploratory behavior policy to collect experience.

The agent may choose a random action during exploration, but the target still uses the maximum next-state Q-value.

This distinction is central to the difference between Q-Learning and SARSA.

10.3 Q-Learning training loop

The essential update is:

action = epsilon_greedy(state, Q, epsilon)

next_state, reward, terminated, truncated, _ = (
    env.step(action)
)

if terminated:
    td_target = reward
else:
    td_target = (
        reward
        + gamma * np.max(Q[next_state])
    )

Q[state, action] += alpha * (
    td_target - Q[state, action]
)

The surrounding loop resets the environment for each episode, accumulates the reward, counts the steps, and stops when the episode terminates or is truncated.

Unlike SARSA, Q-Learning does not need to select the next action to calculate its TD target.

10.4 Training configuration

The recorded Q-Learning experiment used the same nominal parameters as SARSA:

Parameter

Value

Episodes

500

Learning rate (\alpha)

0.5

Discount factor (\gamma)

1.0

Exploration rate (\epsilon)

0.1

The same environment and hyperparameters make the comparison informative, although stochastic training trajectories can still differ.

10.5 Q-Learning training results

Metric

Result

Mean reward, first 50 episodes

-114.62

Mean reward, last 50 episodes

-54.40

Mean steps, last 50 episodes

16.78

The mean reward improved by approximately 60.22 points between the first and last 50-episode windows.

The episode-length moving average declined substantially during training. However, reward remained more negative and more variable than in the SARSA training results.

The reward curve also showed large negative spikes, consistent with episodes that incurred substantial costs, potentially including repeated cliff falls.

These training metrics alone do not establish which learned policy is better.

10.6 Q-Learning greedy evaluation

The learned Q-table was evaluated over 1,000 episodes without epsilon-based exploration.

Observed results:

Metric

Q-Learning result

Evaluation episodes

1,000

Mean reward

-13.0

Mean steps

13.0

Mean cliff falls per episode

0.0

Success rate

100%

Timeout episodes

0

The agent reached the goal in every evaluation episode, never fell off the cliff, and required exactly 13 moves on average.

10.7 Q-Learning learned path

The inspected trajectory was:

[36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47]

The agent:

Moves up from State 36 to State 24.

Travels right along the row above the cliff, from State 24 to State 35.

Moves down from State 35 to the goal at State 47.

This route contains 13 moves and avoids the cliff entirely.

Its total reward is:

[
G=-13
]

This matches the shortest safe path in the environment.

10.8 Q-Learning interpretation

In this experiment, Q-Learning learned the shortest safe path after 500 training episodes.

The greedy policy achieved a 100% success rate over 1,000 evaluation episodes, with no cliff falls or timeouts.

The result demonstrates that Q-Learning successfully learned a policy that is both safe and efficient in this particular experiment.

It does not imply that Q-Learning will always outperform SARSA. Results depend on the environment, learning parameters, exploration strategy, initialization, and training experience.

Part V — SARSA vs. Q-Learning

11. Algorithmic Comparison

The essential difference is the next-state term in the update.

SARSA

[
Q(s,a)\leftarrow Q(s,a)+\alpha
[r+\gamma Q(s',a')-Q(s,a)]
]

SARSA uses the Q-value of the action actually selected in the next state.

Q-Learning

[
Q(s,a)\leftarrow Q(s,a)+\alpha
[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
]

Q-Learning uses the maximum Q-value available in the next state.

Property

SARSA

Q-Learning

Learning type

TD control

TD control

Learned function

(Q(s,a))

(Q(s,a))

Policy relationship

On-policy

Off-policy

Next-state target

(Q(s',a'))

(\max_{a'}Q(s',a'))

Requires selecting next action for target

Yes

No

Can learn from exploratory behavior

Yes

Yes

Uses the next action actually selected

Yes

No

Uses the maximum next-state Q-value

No, unless the selected action is greedy

Yes

Both methods bootstrap and learn from experience without requiring a complete model of the environment.

12. Final Evaluation Comparison

The following table compares the greedy policies obtained after training.

Evaluation metric

SARSA

Q-Learning

Training episodes

500

500

Evaluation episodes

1,000

1,000

Mean reward

-17.0

-13.0

Mean steps

17.0

13.0

Mean cliff falls per episode

0.0

0.0

Success rate

100%

100%

Timeout episodes

0

0

Inspected path length

17

13

Shortest safe path achieved

No

Yes

Main finding

Both algorithms learned policies that safely reached the goal in all recorded evaluation episodes.

However:

SARSA learned a safe 17-step path.

Q-Learning learned the 13-step shortest safe path.

Q-Learning therefore achieved a higher mean reward in the recorded greedy evaluation:

[
-13>-17
]

The difference is entirely consistent with the four additional moves taken by the inspected SARSA path, with both paths avoiding the cliff.

Why did Q-Learning have worse training reward but better evaluation reward?

The final training windows produced:

Training metric

SARSA

Q-Learning

Mean reward, first 50 episodes

-111.52

-114.62

Mean reward, last 50 episodes

-22.90

-54.40

Mean steps, last 50 episodes

20.92

16.78

At first, these results appear inconsistent with the final evaluation.

However, training and evaluation measure different behavior:

During training, both agents used (\epsilon=0.1), so they continued to explore.

Exploratory actions can lead to longer routes and cliff penalties.

During evaluation, the agents selected greedy actions and did not update their Q-tables.

Q-Learning's learned greedy policy was more efficient in this experiment even though its exploratory training episodes had a lower average reward.

Therefore, the last 50 training episodes should not be treated as a direct substitute for greedy policy evaluation.

The evaluation results provide the clearer comparison of the policies learned in these runs.

13. Visualizing Learned Paths

The notebooks visualize the learned routes on the 4-by-12 grid.

The comparison plot shows the following paths:

SARSA

36 → 24 → 12 → 0
   → 1 → 2 → 3 → ... → 11
   → 23 → 35 → 47

This route travels along the top row before returning down to the goal.

Path length: 17 moves.

Q-Learning

36 → 24 → 25 → 26 → ... → 35 → 47

This route travels across the row immediately above the cliff and then moves down into the goal.

Path length: 13 moves.

The visualization highlights an important result: both agents learned to avoid the cliff, but they learned different safe routes.

The route comparison is specific to the inspected greedy policies from these experiments.

Part VI — Implementation Notes

14. Gymnasium API

The control experiments use:

env = gym.make("CliffWalking-v1")

The version CliffWalking-v1 should be used with the Gymnasium installation in this project; CliffWalking-v0 is deprecated in that installation.

The environment returns five values from step():

next_state, reward, terminated, truncated, info = (
    env.step(action)
)

These values have distinct meanings:

next_state: state after the transition.

reward: immediate reward.

terminated: whether the task reached a terminal state.

truncated: whether the episode stopped because of an external limit, such as a time limit.

info: additional environment information.

The distinction between terminated and truncated matters for TD targets.

For a true terminal transition, the target should not include a future-state value:

[
\text{Target}=r
]

For a nonterminal transition, the bootstrap term can be included:

[
\text{Target}=r+\gamma V(s')
]

or, for Q-Learning:

[
\text{Target}=r+\gamma\max_{a'}Q(s',a')
]

The precise treatment of time-limit truncation should match the intended task semantics. A truncated episode is not necessarily a true terminal state.

15. Learning Rate and Discount Factor

Learning rate (\alpha)

The learning rate controls how much a new TD error changes the current estimate.

A very small learning rate can slow learning.

A large learning rate can produce unstable updates.

A value of 0.5 was used in both control experiments.

Discount factor (\gamma)

The discount factor controls how much future rewards contribute to the target.

(\gamma=0) considers only immediate reward.

Values closer to 1 place more weight on future rewards.

The control experiments used gamma=1.0, which is appropriate for this finite episodic task.

Exploration rate (\epsilon)

The exploration rate determines the probability of random action selection during training.

The experiments used:

epsilon = 0.1

This means that approximately 10% of action selections use the exploration branch, although the exact number varies from run to run.

Exploration remains active throughout the recorded training experiments because epsilon was fixed rather than decayed.

16. Evaluation vs. Training

Training and evaluation should be kept separate.

During training:

The Q-table changes.

The agent explores.

Reward and episode length include the effects of exploratory behavior.

During greedy evaluation:

The trained Q-table is fixed.

The agent selects an action with the highest Q-value.

No Q-value updates are performed.

Performance is measured over multiple episodes.

The evaluation code also records cliff falls, success rate, timeouts, and trajectories.

This separation makes it possible to evaluate the learned policy without mixing in ongoing learning updates.

17. Reproducibility and Limitations

The recorded experiments used NumPy seeding and action-space seeding during training:

np.random.seed(42)
env.action_space.seed(42)

A fixed seed helps make experiments easier to reproduce, but it does not guarantee identical results across all code changes, library versions, or random-number usage.

The observed results are from the recorded runs and should not be interpreted as universal performance guarantees.

Additional limitations include:

The control experiments used 500 training episodes.

Epsilon was fixed at 0.1 rather than decayed.

Greedy evaluation used 1,000 episodes.

The evaluation paths and metrics reflect the learned Q-tables from those particular runs.

The TD(0) versus Monte Carlo RMSE comparison is based on the plotted experiment; exact final RMSE values were not recorded in the summarized results.

A more extensive study could evaluate multiple independent seeds and report mean performance and variability across runs.

Part VII — Overall Conclusions

18. What We Learned

This project demonstrated three important stages of TD learning.

TD(0) Prediction

TD(0) learns state values by combining an observed reward with the estimated value of the next state.

The RMSE experiment illustrated incremental learning and showed that TD(0) and Monte Carlo can exhibit different learning curves.

SARSA

SARSA learns action values using the next action selected by its current behavior policy.

In the recorded experiment, it learned a safe policy with a 100% evaluation success rate, zero cliff falls, and a 17-step inspected route.

Q-Learning

Q-Learning learns action values using the maximum next-state Q-value.

In the recorded experiment, it learned the 13-step shortest safe route and achieved a 100% evaluation success rate without cliff falls or timeouts.

Final comparison

The results highlight the difference between the two TD control algorithms:

SARSA learns using the actions selected by its behavior policy.

Q-Learning learns using the greedy next-state value, even while its behavior policy explores.

Both approaches can learn successful policies. In our particular experiment, Q-Learning found a shorter safe route after the same nominal number of training episodes.

The central lesson is that the update rule, exploration strategy, and evaluation procedure all influence the observed performance of a reinforcement learning agent.

19. Next Steps

Possible extensions include:

Compare SARSA and Q-Learning across multiple random seeds.

Introduce epsilon decay and study the exploration-exploitation trade-off.

Test different learning rates.

Plot training reward and episode length for both algorithms on the same axes.

Compare learned policies and Q-tables across additional environments.

Progress to function approximation and Deep Q-Networks (DQN), where a neural network estimates action values instead of storing them in a table.

These extensions build on the prediction, TD error, bootstrapping, and control concepts introduced in this project.