import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding


class TdhEnv(gym.Env):
    def __init__(self):
        self.rewards_setting = [
                                [20, 20],   # MUTUAL COOP
                                [24, -6], # DECEIT
                                [-6, 24], # DECEIT
                                [-2, -2]  # MUTUAL DECEIT
                                ]
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.int)  # DQN
        # self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float64)  # DDPG
        # observation space, czyli zakres wartości, które przekazuję sieciom agentów jako input w pliku simultaneous.py (funkcja forward)
        # [actions, step, enemy_hive, enemy_rewards, my_index, enemy_index]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -6, 0, 0]), high=np.array([1, 20, 9, 24, np.inf, np.inf]), dtype=np.int)
        self.steps_counter = -1
        self.episodes_counter = 1
        self.game = []
        self.rewards = [0, 0]
        self.observation = [random.randint(0, 1), random.randint(0, 1), self.steps_counter, [0, 0]]
        self.rewards_list = []
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        if self.episodes_counter == 1 and self.steps_counter == 0:
            print("Rewards setting:", self.rewards_setting)

        action_0 = actions[0]
        action_1 = actions[1]
        print("act0 before change:", action_0, "act1 before change:", action_1)
        action_0 = 1 if action_0 > 0 else 0
        action_1 = 1 if action_1 > 0 else 0
        print("act0 after change:", action_0, "act1 after change:", action_1)

        if action_0 == 0 and action_1 == 0:
            rewards = self.rewards_setting[0]
        if action_0 == 1 and action_1 == 0:
            rewards = self.rewards_setting[1]
        if action_0 == 0 and action_1 == 1:
            rewards = self.rewards_setting[2]
        if action_0 == 1 and action_1 == 1:
            rewards = self.rewards_setting[3]

        self.game.append([action_0, action_1])
        self.steps_counter += 1
        self.observation = [action_1, action_0, self.steps_counter, rewards]

        if self.steps_counter == 19:
            player_zero = ""
            player_one = ""
            for i in self.game:
                player_zero += str(i[0])
                player_one += str(i[1])
            print(player_zero)
            print(player_one)
            self.episodes_counter += 1
            return self.observation, rewards, 1, {}
        else:
            return self.observation, rewards, 0, {}

    def reset(self):
        self.steps_counter = -1
        self.game = []
        # self.done = 0
        self.observation = [random.randint(0, 1), random.randint(0, 1), self.steps_counter, [0, 0]]
        return self.observation

    def _render(self, mode='human', **kwargs):
        pass
