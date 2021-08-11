from rl.core import Agent
import random


class SuspiciousTitForTatAgent(Agent):
    """
    NPC Suspicious Tit For Tat Agent
    Defects on the first round and imitates its opponent's previous move thereafter.
    """
    def __init__(self, nb_actions):
        super(SuspiciousTitForTatAgent, self).__init__()
        self.compiled = False
        self.step = 0
        self.nb_actions = nb_actions
        # State.
        self.reset_states()

    def compile(self, optimizer, metrics=[]):
        self.compiled = True

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None

    def forward(self, observation):
        # Set step number.
        self.step = observation[0]

        # Select an action.
        if self.step == -1:
            action = self.nb_actions*1
        else:
            action = self.nb_actions*observation[2]

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
