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
        self.epsilon_history = []

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
        
        self.epsilon_history.append(
            self.epsilon
        )

        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)

        max_value = np.max(self.values)

        candidates = np.where(
            self.values == max_value
        )[0]

        return np.random.choice(
            candidates
        )
       

class EpsilonDecayAgent(BanditAgent):
    def __init__(
        self,
        n_arms,
        epsilon=1.0,
        decay=0.995,
        min_epsilon=0.01
    ):
        super().__init__(
            n_arms
        )

        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon

    def select_action(self):
        self.epsilon_history.append(
            self.epsilon
        )
        
        if np.random.random() < self.epsilon:
            action = np.random.randint(
                self.n_arms
            )
        else:
            max_value = np.max(
                self.values
            )
            candidates = np.where(
                self.values == max_value
            )[0]
            action = np.random.choice(
                candidates
            )

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.decay
        )

        return action


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


class UCBAgent(BanditAgent):

    def __init__(
        self,
        n_arms,
        c=2
    ):
        super().__init__(
            n_arms
        )

        self.c = c
        self.total_steps = 0


    def select_action(self):

        self.total_steps += 1


        for arm in range(self.n_arms):

            if self.counts[arm] == 0:
                return arm


        confidence = self.c * np.sqrt(
            np.log(self.total_steps)
            /
            self.counts
        )


        ucb_values = (
            self.values
            +
            confidence
        )


        return np.argmax(
            ucb_values
        )