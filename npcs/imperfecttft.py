from rl.core import Agent
import random


class ImperfectTitForTatAgent(Agent):
    """
    NPC Imperfect Tit For Tat
    Imitates opponent's last move with high (but less than one) probability.
    """
    def __init__(self):
        super(ImperfectTitForTatAgent, self).__init__()
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
        self.step = observation[1]

        # Select an action.
        if self.step == -1:
            action = -1
        else:
            if random.randint(0, 100) > 10:
                action = observation[0]
            else:
                if observation[0] == 1:
                    action = -1
                if observation[0] == 0:
                    action = 1


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
