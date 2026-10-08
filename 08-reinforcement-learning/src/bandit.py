import numpy as np


class MultiArmedBandit:

    def __init__(
        self,
        n_arms=10
    ):

        self.n_arms = n_arms

        self.means = np.random.normal(
            0,
            1,
            n_arms
        )

    def step(
        self,
        action
    ):
        return np.random.normal(
            self.means[action],
            1
        )
      
        
class BanditAgent:
    def __init__(
        self,
        n_arms
    ):
        self.n_arms = n_arms
        self.counts = np.zeros(
            n_arms
        )
        self.values = np.zeros(
            n_arms
        )

    def update(
        self,
        action,
        reward
    ):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] = (
            value +
            (reward - value) / n
        )
    

class EpsilonGreedyAgent(BanditAgent):
    def __init__(
        self,
        n_arms,
        epsilon=0.1
    ):
        super().__init__(
            n_arms
        )

        self.epsilon = epsilon

    def select_action(self):

        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)

        max_value = np.max(self.values)

        candidates = np.where(
            self.values == max_value
        )[0]

        return np.random.choice(
            candidates
        )
       
        
# def run_agent(
#     bandit,
#     agent,
#     steps=1000
# ):
#     rewards = []
#     for _ in range(steps):
#         action = agent.select_action()
#         print("action: ", action)
#         reward = bandit.step(action)
#         print("reward: ", reward)
#         agent.update(action, reward)
#         print("agent: ", agent)
#         rewards.append(reward)
#     return rewards

# bandit = MultiArmedBandit(n_arms=10)
# agent = EpsilonGreedyAgent(n_arms=10, epsilon=0.1)

# rewards = run_agent(
#     bandit,
#     agent,
#     steps=5000
# )