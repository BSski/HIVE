from rl.core import Agent
import random


class AlwaysCooperateAgent(Agent):
    """
    NPC Always Cooperate Agent
    Cooperates unconditionally.
    """
    def __init__(self):
        super(AlwaysCooperateAgent, self).__init__()
        self.compiled = False
        self.step = 0
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
        action = 0

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
