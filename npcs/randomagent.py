from rl.core import Agent
import random


class RandomAgent(Agent):
    """
    NPC Random Agent
    Picks random action.
    """
    def __init__(self):
        super(RandomAgent, self).__init__()
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
        action = random.randint(0,1)

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
