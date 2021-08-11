import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding


class TdhEnv(gym.Env):
    def __init__(self):
        self.rewards_setting = [
                                [20, 20],   # MUTUAL COOP
                                [24, -24],  # DECEIT
                                [-24, 24],  # DECEIT
                                [0, 0]      # MUTUAL DECEIT
                                ]
        self.nb_actions = 10
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([self.nb_actions-1]), dtype=np.int)

        # [step, my_action, enemy_action, my_hive, enemy_hive, my_reward, my_index, enemy_index]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, -6, 0, 0]), high=np.array([20, self.nb_actions-1, self.nb_actions-1, 9, 9, 24, 25, 25]), dtype=np.int)
        self.steps_counter = -1
        self.episodes_counter = 1
        self.game = []
        self.observation = [random.randint(0, 1), random.randint(0, 1), self.steps_counter, [0, 0]]
        self.rewards_list = []
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        action_0 = actions[0]
        action_1 = actions[1]

        action_0 = action_0*(1/self.nb_actions)
        action_1 = action_1*(1/self.nb_actions)

        action_0 = 1 if action_0 > 0.5 else 0
        action_1 = 1 if action_1 > 0.5 else 0

        if action_0 == 0 and action_1 == 0:
            rewards = self.rewards_setting[0]
        elif action_0 == 1 and action_1 == 0:
            rewards = self.rewards_setting[1]
        elif action_0 == 0 and action_1 == 1:
            rewards = self.rewards_setting[2]
        elif action_0 == 1 and action_1 == 1:
            rewards = self.rewards_setting[3]


        self.game.append([action_0, action_1])
        self.steps_counter += 1
        self.observation = [action_1, action_0, self.steps_counter, rewards]

        if self.steps_counter == 19:
            player_zero_moves = ""
            player_one_moves = ""
            for i in self.game:
                player_zero_moves += str(i[0])
                player_one_moves += str(i[1])
            self.episodes_counter += 1
            return self.observation, rewards, 1, {}
        else:
            return self.observation, rewards, 0, {}

    def reset(self):
        self.steps_counter = -1
        self.game = []
        self.observation = [random.randint(0, 1), random.randint(0, 1), self.steps_counter, [0, 0]]
        return self.observation

    def _render(self, mode='human', **kwargs):
        pass
