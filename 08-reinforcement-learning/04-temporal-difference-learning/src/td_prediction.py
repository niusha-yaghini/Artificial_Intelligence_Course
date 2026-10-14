import numpy as np

class RandomWalk:
    def __init__(self):
        self.state = 2

    def reset(self):
        self.state = 2
        return self.state

    def step(self, action):
        if action == "left":
            self.state -= 1
        else:
            self.state += 1

        reward = 0
        done = False

        if self.state == 0:
            done = True
        elif self.state == 4:
            reward = 1
            done = True

        return self.state, reward, done

 