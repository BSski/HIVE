from rl.core import Agent
import random


class SuspiciousTitForTatAgent(Agent):
    """
    NPC Suspicious Tit For Tat Agent
    Defects on the first round and imitates its opponent's previous move thereafter.
    """
    def __init__(self):
        super(SuspiciousTitForTatAgent, self).__init__()
        self.counter = -1
        self.action_next = None
        self.compiled = False
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

        if self.counter == 0:
            action = 1
        else:
            action = observation[0]

        if self.counter == 20:
            self.counter = -1

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
