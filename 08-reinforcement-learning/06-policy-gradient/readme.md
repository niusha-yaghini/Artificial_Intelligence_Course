Policy Gradient Methods: REINFORCE and REINFORCE with Baseline
This section implements two policy-gradient algorithms from scratch with
PyTorch and Gymnasium's CartPole-v1: vanilla REINFORCE and REINFORCE
with a learned value-function baseline. The experiments demonstrate
stochastic policy learning, discounted returns, policy-gradient losses,
value estimation, advantage calculation, training instability, and
independent evaluation.
Contents
- Learning objectives
- Environment
- Policy-gradient foundations
- Policy network and action
  sampling
- Episode collection and returns
- Vanilla REINFORCE
- REINFORCE results
- REINFORCE with Baseline
- Baseline results
- Evaluation
- Comparison and interpretation
- Implementation notes
- Limitations
- Reproduction
- Conclusion
Learning objectives
The experiments cover: - the difference between value-based methods such
as DQN and policy-based methods; - a stochastic policy represented by
action probabilities; - sampling actions with PyTorch's Categorical
distribution; - discounted returns; - the REINFORCE policy-gradient
loss; - a value network approximating (V(s)); - the advantage estimate
(A_t = G_t - V(s_t)); - separate optimizers for the policy and value
networks; - learning curves, moving averages, and evaluation without
gradient updates.
Environment
The experiments use Gymnasium's CartPole-v1.
Observations and actions
The observation is a continuous vector of four values: 1. cart position;
2. cart velocity; 3. pole angle; 4. pole angular velocity.
The observation space is a four-dimensional Box, so the state input
has shape (4,).
The action space is Discrete(2): - 0: push left; - 1: push right.
Rewards and episode termination
The environment gives a reward of +1 for each step in which the
episode continues. An episode ends when the cart-pole fails or when the
CartPole-v1 time limit is reached. The maximum episode length is 500
steps, so total episode reward is also the episode length and is capped
at 500.
A commonly used CartPole-v1 reference is a mean reward of at least 475
over 100 evaluation episodes. Reaching 500 in some training episodes is
not the same as meeting this evaluation criterion.
Policy-gradient foundations
Value-based versus policy-based methods
DQN approximates action values (Q(s,a)) and commonly selects the action
with the largest estimated value. Policy-gradient methods instead learn
a policy directly:
\(`\pi`{=tex}\_`\theta`{=tex}(a `\mid `{=tex}s)\)
The policy maps a state to a probability distribution over actions. In
CartPole, it produces probabilities for the left and right actions.
Logits and probabilities
The policy network outputs two raw values called logits. A softmax
can convert them to probabilities:
`\mid `{=tex}s)=`\frac{e^{z_i}}{\sum_j e^{z_j}}`{=tex} \]

where (z_i) is the logit for action (i). The implementation passes
logits directly to `Categorical(logits=logits)`, which handles the
conversion stably.

### Stochastic action selection

Actions are sampled from the policy distribution:

\[ a_t
`\sim `{=tex}`\pi`{=tex}\_`\theta`{=tex}(`\cdot `{=tex}`\mid `{=tex}s_t)
This is different from always choosing the most probable action. Greedy
action selection uses the maximum-probability action; stochastic
selection samples from the learned distribution.
Policy network and action sampling
Architecture
The policy network is:
Input: 4 state features
Linear(4, 128)
ReLU
Linear(128, 128)
ReLU
Linear(128, 2)
Output: logits for actions 0 and 1
The final layer has no softmax. Categorical consumes the logits
directly.
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.network(state)
Sampling an action
logits = policy_net(state_tensor)
distribution = Categorical(logits=logits)
action = distribution.sample()
log_prob = distribution.log_prob(action)
action is passed to the environment. log_prob is retained for the
policy-gradient update and remains connected to the network's
computation graph.
For an action with probability (p), its log probability is
(\log{=tex}(p)), which is non-positive because (0 < p
\leq 1{=tex}).
Episode collection and returns
At each step, the episode-generation function stores: - the current
state; - the log probability of the selected action; - the reward
received.
The baseline version also needs the states to train its value network.
Discounted return
The return from time step (t) is:
\(G_t = `\sum`{=tex}*{k=0}^{T-t-1}`\gamma`{=tex}^k r*{t+k}\)
where (G_t) is the discounted return, (r_t) is the immediate reward,
(\gamma{=tex}) is the discount factor, and (T) is the episode length.
Both experiments use:
\(`\gamma `{=tex}= 0.99\)
Returns are calculated backward using:
\(G_t = r_t + `\gamma `{=tex}G\_{t+1}\)
For an episode with reward 1.0 at every step, early steps have larger
returns because more future rewards remain. The final return is 1.0.
Vanilla REINFORCE
REINFORCE is a Monte Carlo policy-gradient algorithm. It collects a
complete episode, calculates returns, and then updates the policy.
Policy objective and loss
The policy-gradient objective can be expressed as:
J(`\theta`{=tex})=`\mathbb{E}`{=tex}`\left[G_t\log\pi_\theta(a_t\mid s_t)\right]`{=tex}\]

PyTorch optimizers minimize a loss, so the implementation minimizes the
negative sum:

\[ L\_{`\text{policy}`{=tex}}(`\theta`{=tex})=
-`\sum`{=tex}*{t=0}\^{T-1}G_t`\log`{=tex}`\pi`{=tex}*`\theta`{=tex}(a_t`\mid `{=tex}s_t)
Simplified implementation:
log_probs_tensor = torch.stack(log_probs)
returns_tensor = torch.tensor(returns, dtype=torch.float32)

policy_loss = -(log_probs_tensor * returns_tensor).sum()

optimizer.zero_grad()
policy_loss.backward()
optimizer.step()
Training procedure
For each episode: 1. interact with the environment until termination or
truncation; 2. store action log probabilities and rewards; 3. calculate
discounted returns; 4. compute the policy loss; 5. backpropagate and
update the policy network; 6. record the episode reward and loss.
The implementation updates the policy once per complete episode.
Why vanilla REINFORCE can be unstable
The raw return weights each log-probability term. Returns can vary
significantly between episodes and across time steps, producing
high-variance gradient estimates. A policy may improve rapidly, but
later updates can also degrade a previously strong policy.
Vanilla REINFORCE configuration
  Setting                                                     Value
  Environment                                         CartPole-v1
  Training episodes                                           1,000
  State dimension                                                 4
  Action dimension                                                2
  Hidden layers                                        [128, 128]
  Activation                                                   ReLU
  Optimizer                                                    Adam
  Learning rate                                             0.001
  Discount factor (gamma)                                  0.99
  Policy loss                   Negative sum of log_prob * return
  Progress interval                                     50 episodes
  Moving-average window                                 50 episodes
REINFORCE results
The observed training output was:
Episode   Mean reward, last 50   Mean loss, last 50
     50                  35.00               471.57
    100                  42.98               650.79
    150                  68.24             1,372.76
    200                 180.54             5,834.62
    250                 233.56             8,067.53
    300                 125.92             3,017.93
    350                 153.80             3,970.24
    400                 257.18             8,322.48
    450                 220.76             7,019.16
    500                 333.82            12,232.04
    550                 308.82            10,909.63
    600                 250.58             8,071.18
    650                 253.90             7,938.88
    700                 324.36            11,241.23
    750                 148.24             3,432.54
    800                 148.26             3,352.72
    850                 238.42             7,115.31
    900                 497.92            18,904.94
    950                 353.76            12,573.17
  1,000                  79.76             1,437.35
Summary
  Metric                                          Result
  Mean reward, first 50 episodes                 35.00
  Mean reward, last 50 episodes                  79.76
  Best 50-episode moving average                500.00
  Best moving-average window ending at     Episode 902
The moving average reached the maximum reward around episode 900, but
performance then deteriorated sharply. The final 50-episode mean was
only 79.76. This demonstrates that the best training window and the
final training window can tell very different stories.
The loss also varied sharply and reached large values. This is not by
itself proof that learning was worsening: the loss is summed across time
steps, so its scale depends on episode length, returns, and log
probabilities. Reward and independent evaluation are more informative
measures of policy performance.
REINFORCE with Baseline
The baseline version adds a value network to estimate the expected
return from a state.
Value function and network
The value network approximates:
V\_`\phi`{=tex}(s)`\approx`{=tex}`\mathbb{E}`{=tex}\[G_t`\mid `{=tex}S_t=s\]
Its architecture is:
Input: 4 state features
Linear(4, 128)
ReLU
Linear(128, 128)
ReLU
Linear(128, 1)
Output: estimated state value V(s)
Simplified implementation:
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)
Advantage
The baseline version replaces the raw return with an advantage estimate:
\(A_t=G_t-V\_`\phi`{=tex}(s_t)\)
Advantage measures whether the observed return was better or worse than
the value network expected: - positive advantage: return was better than
estimated; - negative advantage: return was worse than estimated; -
near-zero advantage: return was close to the estimate.
A state-dependent baseline can reduce the variance of policy-gradient
estimates without changing their expected gradient, provided the
baseline does not introduce action-dependent bias.
Policy loss with baseline
\[
-`\sum`{=tex}*t`\log`{=tex}`\pi`{=tex}*`\theta`{=tex}(a_t`\mid `{=tex}s_t)A_t
\]
The implementation detaches the value estimate while calculating the
policy advantage, so policy-loss gradients do not update the value
network:
values = value_net(states_tensor)
advantages = returns_tensor - values.detach()
policy_loss = -(log_probs_tensor * advantages).sum()
Value loss
The value network is trained to predict observed returns using mean
squared error:
\[
L_V(`\phi`{=tex})=`\frac{1}{T}`{=tex}`\sum`{=tex}*t(V*`\phi`{=tex}(s_t)-G_t)\^2
\]
value_loss = nn.functional.mse_loss(values, returns_tensor)
The policy and value networks use separate Adam optimizers. The policy
network is updated using the policy loss, while the value network is
updated using the value MSE loss.
Training procedure
For each complete episode: 1. record states, log probabilities, and
rewards; 2. calculate discounted returns; 3. predict state values; 4.
calculate detached advantages; 5. calculate the policy loss and value
MSE loss; 6. update the policy network; 7. update the value network; 8.
record the reward and both losses.
Baseline configuration
  Setting                                                        Value
  Environment                                            CartPole-v1
  Training episodes                                              1,000
  Policy hidden layers                                    [128, 128]
  Value hidden layers                                     [128, 128]
  Activation                                                      ReLU
  Policy optimizer                                                Adam
  Value optimizer                                                 Adam
  Policy learning rate                                         0.001
  Value learning rate                                          0.001
  Discount factor (gamma)                                     0.99
  Policy loss                   Negative sum of log_prob * advantage
  Value loss                                        Mean squared error
  Progress interval                                        50 episodes
  Moving-average window                                    50 episodes
No advantage normalization, gradient clipping, entropy bonus, or
learning-rate schedule was used in the recorded implementation.
Baseline results
The observed training output was:
        Episode Mean reward, last Mean policy loss,  Mean value loss,
                               50           last 50           last 50
             50             26.46            228.07            185.33

            100             56.96            550.47            380.47

            150            140.18          1,002.80            707.01

            200            149.84            779.53            717.20

            250            257.60          1,662.70            801.67

            300            261.54          1,189.03            367.62

            350            459.94            293.18            383.19

            400            419.82            -88.98            625.44

            450            479.26            168.06            553.93

            500            467.22            108.66            482.04

            550            492.86            369.78            472.02

            600            476.18            100.14            391.29

            650            495.58           -235.11            487.01

            700            446.44            545.01            477.07

            750            384.10             70.65            311.89

            800            497.58           -243.55            430.73

            850            496.70           -321.61            550.16

            900            270.72            391.39            301.83

            950            279.46            182.21            228.01

          1,000            495.48            337.15            561.91
Summary
  Metric                                          Result
  Mean reward, first 50 episodes                 26.46
  Mean reward, last 50 episodes                 495.48
  Best 50-episode moving average                500.00
  Best moving-average window ending at     Episode 535
The baseline version reached high performance earlier in this run. It
maintained a high moving average for long stretches but was not
perfectly stable: the mean reward fell to 270.72 at episode 900 and
279.46 at episode 950, then recovered to 495.48 in the final 50
episodes.
A negative policy loss is possible because the advantage may be
negative. It is not automatically an error. The policy loss and value
MSE have different definitions and scales and should not be directly
compared.
Evaluation
The trained baseline policy was evaluated for 100 episodes. The
evaluation function used policy_net.eval() and torch.no_grad(),
performed no backward pass or optimizer step, and used episode seeds
starting at 20,000.
Two action-selection modes were tested.
- Greedy: select the action with the largest logit. This is
  equivalent to selecting the action with the largest policy
  probability.
- Stochastic: sample an action from Categorical(logits=logits),
  matching the action-selection method used during training.
Evaluation results
  Metric                                  Greedy    Stochastic
  Evaluation episodes                        100           100
  Mean reward                           500.00      500.00
  Median reward                         500.00      500.00
  Standard deviation                      0.00        0.00
  Minimum reward                           500         500
  Maximum reward                           500         500
  Episodes reaching 500 steps        100 / 100   100 / 100
  Episodes reaching at least 475     100 / 100   100 / 100
The trained baseline policy reached the 500-step time limit in all 100
greedy evaluation episodes and all 100 stochastic evaluation episodes.
Both means equal the environment's maximum possible reward. The
stochastic result is particularly notable because the policy was allowed
to sample actions instead of always choosing the most probable action.
These results describe the tested model and seed set. They do not
guarantee perfect performance under every possible initial condition.
Comparison and interpretation
  Metric                                              REINFORCE   REINFORCE with Baseline
  Training episodes                                       1,000                     1,000
  Mean reward, first 50 episodes                        35.00                   26.46
  Mean reward, last 50 episodes                         79.76                  495.48
  Best 50-episode moving average                       500.00                  500.00
  Window ending at best moving average            Episode 902             Episode 535
  Independent 100-episode evaluation recorded                No                       Yes
  Greedy evaluation mean                           Not recorded                  500.00
  Stochastic evaluation mean                       Not recorded                  500.00
Main findings
1. The baseline version reached strong performance earlier in this
   run. Its best 50-episode moving average reached 500 around episode
   535, compared with around episode 902 for vanilla REINFORCE.
2. The baseline version finished training much more strongly. Its
   final 50-episode mean was 495.48, compared with 79.76 for
   vanilla REINFORCE.
3. Neither training curve was monotonic. Baseline reduced the
   observed instability but did not prevent sharp performance drops.
4. The baseline evaluation was excellent. Both greedy and
   stochastic evaluation reached 500 in all 100 episodes.
5. Vanilla REINFORCE was not independently evaluated in the recorded
   experiment. Therefore, the two models' independent evaluation
   performance cannot be directly compared.
The results favor REINFORCE with Baseline in these particular runs, but
only one training run per method was recorded. This is not proof that
the baseline method always outperforms vanilla REINFORCE.
Implementation notes
Gymnasium API
The code uses:
state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
An episode ends when either terminated or truncated is true.
Tensor shapes
For an episode with (T) steps: - states: (T, 4); - stacked log
probabilities: (T,); - rewards: length T; - returns: (T,); - value
predictions: (T,); - advantages: (T,).
These lengths must align so each log probability, return, value
estimate, and advantage refers to the same time step.
Why states are copied
States are stored using state.copy() so the episode contains the
observation values at each step rather than references to a mutable
array.
Why the value estimate is detached
values.detach() prevents the policy loss from backpropagating through
the value estimate. The value network is trained separately using its
MSE loss.
Loss-scale caution
Vanilla REINFORCE uses a sum of log_prob * return; baseline REINFORCE
uses a sum of log_prob * advantage; the value network uses mean
squared error. These quantities have different meanings and scales.
Their magnitudes should not be compared as if they were the same metric.
Evaluation does not update weights
Evaluation continues to interact with the environment, but no gradients
or optimizer updates are performed. It measures the policy represented
by the already-trained weights.
Limitations
1. One run per method: outcomes depend on initialization, action
   sampling, and environment trajectories. Multiple independent
   training seeds would provide stronger evidence.
2. No independent evaluation for vanilla REINFORCE: only the
   baseline model has recorded 100-episode evaluation results.
3. Training and evaluation are different: a high training moving
   average does not replace evaluation on fixed seeds.
4. Baseline did not eliminate instability: its training curve still
   showed a pronounced drop near episodes 900--950.
5. The 500-step cap affects interpretation: a reward of 500 means
   the episode reached the time limit, not that the task can continue
   indefinitely.
6. No hyperparameter sweep was performed: the recorded settings
   were not systematically tuned.
7. No universal algorithm ranking follows from these runs:
   conclusions apply to these specific implementations and runs.
A stronger controlled comparison would evaluate both saved policies on
the same fixed seed set, use the same checkpoint-selection procedure,
and repeat training with multiple independent seeds.
Reproduction
Suggested folder structure
06_policy_gradient_methods/
├── 01_policy_network.ipynb
├── README.md
└── results/
    ├── reinforce_baseline_cartpole.pth
    └── reinforce_baseline_results.npz
The notebook name may differ depending on the actual project structure.
The results files correspond to the saved baseline checkpoint and
training/evaluation arrays.
Dependencies
- Python
- NumPy
- PyTorch
- Gymnasium
- Matplotlib
Install packages if needed:
pip install numpy torch gymnasium matplotlib
Run notebook cells in order: create the environment, define the policy
network, inspect logits and action probabilities, implement episode
collection and returns, train vanilla REINFORCE, plot its learning
curves, define the value network, train the baseline version, compare
the curves, evaluate the baseline policy, and save the checkpoint and
metrics.
Results can vary with package versions, seeds, and platform details.
Fixed seeds improve reproducibility but do not guarantee identical
outcomes on every system.
Conclusion
This section implemented vanilla REINFORCE and REINFORCE with Baseline
from scratch in PyTorch on Gymnasium's CartPole-v1.
Vanilla REINFORCE learned to balance the pole for long periods, but its
learning curve fluctuated strongly. Its 50-episode moving average
briefly reached the maximum reward around episode 900, then the final
50-episode mean fell to 79.76.
REINFORCE with Baseline added a learned value function and used the
advantage (G_t - V(s_t)) to weight policy updates. In the recorded run,
it reached a best 50-episode moving average of 500.00 around episode
535 and finished with a final-50 mean of 495.48. It still experienced
a temporary performance collapse, so the baseline improved the observed
learning behavior but did not make training perfectly stable.
The trained baseline policy achieved a mean reward of 500.00 in both
greedy and stochastic evaluation over 100 episodes, with every episode
reaching the 500-step limit.
The main lesson is that a value baseline can provide a useful reference
for policy-gradient updates and substantially improve observed learning
behavior. Training curves, independent evaluation, and careful
experimental controls remain essential when interpreting
reinforcement-learning results.