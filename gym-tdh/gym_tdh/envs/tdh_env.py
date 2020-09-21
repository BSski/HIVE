import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding


class TdhEnv(gym.Env):

    def __init__(self):
        self.plan = "01010101010101010101"
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)
        self.counter = -1
        self.game = []
        self.reward = 0
        self.observation = 1
        self.cash = 200
        self.done = 0
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.observation = 1
        self.counter += 1
        if action == 0 and int(self.plan[self.counter]) == 0:
            reward = 0
        if action == 1 and int(self.plan[self.counter]) == 0:
            reward = 15
        if action == 0 and int(self.plan[self.counter]) == 1:
            reward = -20
        if action == 1 and int(self.plan[self.counter]) == 1:
            reward = 5

        self.cash += reward
        self.observation = int(self.plan[self.counter])
        self.game.append([action, int(self.plan[self.counter])])

        if self.counter >= 19:
            self.done = True
            self.reset()
            return self.observation, reward, 1, action

        return self.observation, reward, self.done, action

    def reset(self):
        self.counter = -1
        self.done = 0
        self.observation = self.np_random.random(1)
        return self.observation

tdh = TdhEnv()
print(tdh.step(0))
for i in range(19):
    print(tdh.step(int(random.choice("01"))))
print(tdh.cash)
