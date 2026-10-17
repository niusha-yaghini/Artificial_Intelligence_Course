Deep Q-Network (DQN) --- CartPole
This project section implements Deep Q-Network (DQN) from scratch with
PyTorch, evaluates the learned policy on Gymnasium's CartPole-v1, and
runs an initial comparison against Stable-Baselines3 DQN.
Contents
- Learning goals
- Environment
- DQN theory
- Custom implementation
- Training results
- Stable-Baselines3 comparison
- Comparison limitations
- Reproduction
- Lessons and conclusion
Learning goals
The goal was to understand and implement the main parts of DQN rather
than only call a ready-made algorithm: - neural-network approximation of
action values; - epsilon-greedy exploration with epsilon decay; -
experience replay; - a separate target network; - mini-batch
optimization and Huber loss; - periodic greedy evaluation and checkpoint
selection; - comparison with a library implementation.
Environment
The environment is CartPole-v1.
The observation is a continuous vector of four values:
1. cart position;
2. cart velocity;
3. pole angle;
4. pole angular velocity.
The action space is Discrete(2):
- 0: push left;
- 1: push right.
Each step gives reward +1. An episode ends when the pole/cart
condition fails or the 500-step time limit is reached. Thus episode
reward equals episode length and is capped at 500.
A commonly used CartPole-v1 benchmark is a mean reward of at least 475
over 100 evaluation episodes. This is different from achieving 500 in
some individual episodes.
DQN theory
Q-value approximation
Tabular Q-learning is impractical for continuous observations. DQN
approximates the action-value function using a neural network:
\(Q(s,a;`\theta`{=tex})\)
The network receives a state and outputs one Q-value for each action.
With two CartPole actions, the output shape is (batch_size, 2). The
greedy action is:
\(a\^\* = `\arg`{=tex}`\max`{=tex}\_a Q(s,a;`\theta`{=tex})\)
Bellman target
The one-step target used by DQN is:
`\gamma`{=tex}(1-d)`\max`{=tex}*{a'}Q*{`\text{target}`{=tex}}(s',a') \]

Here, `r` is the reward, `gamma` is the discount factor, `s'` is the
next state, and `d` marks true environment termination. For a true
terminal transition, the target is just the reward. The implementation
stores `terminated` separately from `truncated`, so a time-limit
truncation does not automatically remove the bootstrap term.

### Loss

The custom implementation uses Smooth L1 (Huber) loss:

\[
L(`\theta`{=tex})=`\operatorname{SmoothL1}`{=tex}(Q(s,a;`\theta`{=tex}),y)
Only the Q-value for the action actually taken is compared to the
target. PyTorch gather() selects that value from the network output.
Experience replay
A transition is stored as:
\((s,a,r,s',`\text{terminated}`{=tex})\)
The replay buffer stores a bounded history of transitions and samples
random mini-batches for training. This allows experiences to be reused
and reduces dependence on the order of consecutive observations. The
buffer is implemented with deque(maxlen=capacity), and state arrays
are copied when stored.
Target network
Two networks are used: - Policy network: trained by gradient
descent. - Target network: used to calculate target Q-values and
held fixed between updates.
Every 500 environment steps, the target network is hard-updated:
target_net.load_state_dict(policy_net.state_dict())
This reduces how quickly the learning target changes as the policy
network is optimized.
Epsilon-greedy exploration
During training, a random action is selected with probability epsilon;
otherwise the agent selects the action with the largest Q-value:
\[
```{=tex}
\begin{cases}
\text{random action}, & \text{with probability } \epsilon\\
\arg\max_a Q(s,a), & \text{otherwise}
\end{cases}
```
\]
The custom implementation uses linear epsilon decay from 1.0 to 0.05
over 50,000 environment steps. Evaluation uses greedy actions with no
random exploration.
Custom implementation
Network architecture
The custom PyTorch network is:
Input: 4 features
Linear(4, 128)
ReLU
Linear(128, 128)
ReLU
Linear(128, 2)
Output: Q-values for left and right
Main hyperparameters
  Parameter                                                 Value
  Environment                                       CartPole-v1
  Final training run                               3,000 episodes
  Hidden layers                                      [128, 128]
  Activation                                                 ReLU
  Optimizer                                                  Adam
  Learning rate                                           0.001
  Discount factor (gamma)                                0.99
  Replay capacity                                        10,000
  Batch size                                                 64
  Loss                                          Smooth L1 / Huber
  Gradient norm clipping                                   10.0
  Epsilon start / end                              1.0 / 0.05
  Epsilon decay duration               50,000 environment steps
  Target update interval                  500 environment steps
  Periodic evaluation                          Every 100 episodes
  Episodes per periodic evaluation                             20
Training step
Each update: 1. samples a random mini-batch from replay memory; 2.
converts arrays to PyTorch tensors; 3. predicts Q-values for the sampled
states; 4. uses gather() to select Q-values for recorded actions; 5.
computes target values with target_net inside torch.no_grad(); 6.
calculates Smooth L1 loss; 7. runs zero_grad(), backward(), gradient
clipping, and optimizer.step().
Only the policy network is optimized. The target network does not
receive gradients.
Training loop and checkpointing
The training loop interacts with the environment, stores transitions,
trains whenever the buffer contains at least one batch, and synchronizes
the target network every 500 environment steps. Every 100 episodes, the
current policy is evaluated greedily over 20 episodes. If the evaluation
mean improves, the policy weights are copied and kept as the best
checkpoint.
At the end of training, the best periodically evaluated weights are
restored. This matters because DQN performance did not improve
monotonically: the best checkpoint was observed before the last training
episode.
Evaluation
The evaluate_dqn function uses: - policy_net.eval(); -
torch.no_grad(); - greedy argmax action selection; - no loss
calculation or optimizer step.
Therefore, network weights are not updated during evaluation. The agent
continues interacting with the environment, but the evaluation measures
the policy rather than training it.
Training results
Several runs were used while developing the implementation:
  Experiment                          Result
  500-episode run                     Last-50 training mean: 125.68;
                                      independent evaluation mean:
                                      109.16
  1,500-episode run                   Best periodic evaluation mean:
                                      341.4; independent evaluation
                                      mean: 300.29
  Final 3,000-episode run             Best periodic evaluation mean:
                                      489.95; independent evaluation
                                  mean: 459.65
The final 3,000-episode run recorded 178,583 gradient updates. Its
best periodic evaluation mean, 489.95, occurred at episode 2,700.
Evaluation results declined at later checkpoints, so the best checkpoint
was restored instead of using the final episode's weights.
The best periodic result was based on 20 episodes and reused evaluation
seeds across checkpoints. It should not be confused with the separate
100-episode independent evaluation.
Independent evaluation of the custom model
The restored custom checkpoint was evaluated for 100 episodes using
seeds starting at 20,000. No gradient updates were performed.
  Metric                                 Result
  Episodes                                  100
  Mean reward                          459.65
  Median reward                         500.0
  Standard deviation                 121.2625
  Minimum reward                           62
  Maximum reward                          500
  Episodes reaching 500              90 / 100
  Episodes reaching at least 475     90 / 100
Reward distribution:
  Reward group     Count
  Exactly 500         90
  475--499             0
  Below 475           10
The ten lowest rewards were:
[62, 72, 74, 80, 98, 105, 110, 111, 114, 139]
This is a bimodal result: 90 episodes reached the 500-step cap, while 10
failed considerably earlier. The median of 500 is excellent, but the
failures pull down the mean. Since the mean 459.65 is below the
commonly used 475 criterion, the accurate conclusion is strong
performance with remaining instability, not that the model passed the
benchmark criterion.
Stable-Baselines3 comparison
A separate notebook, 02_dqn_library_comparison.ipynb, trained the
library DQN using Stable-Baselines3 2.7.0. This separation keeps the
custom implementation readable and avoids retraining it during the
comparison.
Library configuration
  Parameter                                                    Value
  Stable-Baselines3                                          2.7.0
  Algorithm                                  DQN("MlpPolicy", ...)
  Environment                                          CartPole-v1
  Training budget                        178,646 environment steps
  Learning rate                                              0.001
  Replay capacity                                           10,000
  Learning starts                                         64 steps
  Batch size                                                    64
  Gamma                                                       0.99
  Target update interval                                 500 steps
  Training frequency                          Every environment step
  Gradient steps per training event                              1
  Network architecture                                  [128, 128]
  Activation                                                    ReLU
  Initial / final epsilon                             1.0 / 0.05
  Epsilon decay duration                Approximately 50,000 steps
  Model seed                                                   123
The library model was saved as library_dqn_cartpole.zip. Evaluation
used deterministic actions via
library_model.predict(obs, deterministic=True).
Library evaluation
  Metric                    Result
  Episodes                     100
  Mean reward             111.14
  Median reward           111.00
  Standard deviation        7.45
  Minimum reward              98
  Maximum reward             126
  Mean episode length     111.14
Observed comparison
  Metric                  Custom DQN   Library DQN
  Evaluation episodes            100           100
  Mean reward               459.65      111.14
  Median reward             500.00      111.00
  Standard deviation        121.26        7.45
  Minimum reward                62          98
  Maximum reward               500         126
The observed mean difference is 348.51 in favor of the custom model.
This is a result of these specific runs, not proof that custom DQN is
generally better than Stable-Baselines3.
Comparison limitations
This is an initial, not fully controlled comparison. The limitations
should be reported alongside the numbers.
1. Different evaluation seeds: the custom model was evaluated using
   seeds starting at 20,000; the library model used seeds 0 through
   99. Thus, the means were not measured on identical initial
   conditions. A strict paired comparison should evaluate both saved
   models on the same 100 seeds.
2. Different implementation details: similar hyperparameters do not
   guarantee identical replay handling, exploration scheduling,
   optimization timing, or target-network behavior.
3. Training budget and updates differ slightly: custom training
   recorded 178,583 gradient updates; the library run used 178,646
   environment steps and reported 177,977 updates. Environment steps
   and gradient updates are different measures.
4. Only one training run per implementation: results can depend on
   initialization and randomness. Multiple independent training seeds
   are needed for a robust algorithm-level conclusion.
5. Checkpoint selection differs: the custom implementation restored
   its best periodic evaluation checkpoint. The library model was
   evaluated from the final saved model, without equivalent
   best-checkpoint selection.
6. The library model was not tuned: its mean reward was 111.14,
   but this experiment did not investigate or tune the causes of its
   low performance.
The appropriate conclusion is:
In the runs recorded here, the custom DQN achieved a much higher
evaluation reward than the Stable-Baselines3 DQN. However, evaluation
seeds and checkpoint-selection procedures differed, and the library
model was not tuned. These results do not establish that the custom
implementation is inherently superior.

For a stricter comparison, evaluate both saved models on exactly the
same seeds. For stronger evidence, repeat training with several
independent seeds for each implementation.
Reproduction
Suggested project structure
05_deep_q_network/
├── 01_dqn_cartpole.ipynb
├── 02_dqn_library_comparison.ipynb
├── results/
│   ├── custom_dqn_metrics.json
│   ├── custom_dqn_arrays.npz
│   ├── custom_dqn_best.pth
│   ├── library_dqn_cartpole.zip
│   ├── library_dqn_eval.npz
│   └── dqn_mean_reward_comparison.png
└── README.md
The exact files depend on which notebook cells were run and where
outputs were saved.
Dependencies
The experiments used: - NumPy - PyTorch (2.7.0+cpu during the custom
experiment) - Gymnasium - Matplotlib - Stable-Baselines3 2.7.0
Install if needed:
pip install numpy torch gymnasium matplotlib stable-baselines3
Run 01_dqn_cartpole.ipynb to recreate the custom model and
02_dqn_library_comparison.ipynb for the library experiment. The
results/ path is relative to the current working directory; run
notebooks from the project directory or adjust the path.
Results can vary with package versions, seeds, hardware, and environment
behavior. Seeding helps reproducibility but does not guarantee
bit-for-bit identical results across platforms.
Lessons and conclusion
1. DQN is more than a neural network. Replay memory, a target
   network, exploration, and a stable update procedure are core
   components.
2. Training reward differs from evaluation reward. Exploration can
   lower training reward; greedy evaluation measures the learned policy
   without random actions.
3. The last checkpoint is not necessarily the best checkpoint.
   Periodic evaluation found a strong custom policy at episode 2,700,
   followed by lower results. Checkpoint selection preserved the better
   policy.
4. Loss alone does not measure policy quality. DQN's target changes
   over training, so loss need not decrease monotonically.
5. Median can hide failure cases. The custom model reached 500 in
   90% of episodes, but 10 episodes ended early. Both mean and
   distribution matter.
6. Comparison conditions matter. Use the same evaluation seeds and
   clearly report training budgets and checkpoint-selection rules.
7. One run cannot rank algorithms generally. The library result is
   a useful initial comparison, not a universal conclusion.
The custom DQN implementation reached a mean reward of 459.65 over 100
independent episodes and the 500-step limit in 90 episodes. This is
strong performance, but the mean remains below the commonly used 475
criterion and a small set of episodes failed early.
The Stable-Baselines3 run achieved a mean of 111.14 in its recorded
evaluation. The result favors the custom model in these particular
experiments, but the seed sets differed and the library model was not
tuned or selected using an equivalent best-checkpoint process. A paired
evaluation on identical seeds is needed for a stricter comparison.
The main outcome is a complete, understandable DQN training pipeline and
an experiment record that separates measured results from broader claims
about algorithm performance.