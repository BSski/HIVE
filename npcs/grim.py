from rl.core import Agent
import random


class GRIMAgent(Agent):
    """
    NPC GRIM Agent
    Cooperates until its opponent has defected once, and then defects for the rest of the game.
    """
    def __init__(self):
        super(GRIMAgent, self).__init__()
        self.counter = -1
        self.compiled = False
        self.defected_flag = 0
        # State.
        self.reset_states()

    def compile(self, optimizer, metrics=[]):
        self.compiled = True

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None

    def forward(self, observation):
        # Select an action.
        self.counter += 1
        if self.counter > 0:
            if observation[0] == 1:
                self.defected_flag = 1

        if self.defected_flag == 0:
            action = 0
        else:
            action = 1

        if self.counter == 20:
            self.counter = -1
            self.defected_flag = 0

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        metrics = 0, 0, 0
        return metrics

    @property
    def metrics_names(self):
        return "NPC", "NPC", "NPC"
