from rl.core import Agent
import random


class ImperfectTitForTatAgent(Agent):
    """
    NPC Imperfect Tit For Tat
    Imitates opponent's last move with high (but less than one) probability.
    """
    def __init__(self, nb_actions):
        super(ImperfectTitForTatAgent, self).__init__()
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
            action = 0
        else:
            if random.randint(0, 100) > 10:
                action = self.nb_actions*observation[2]
            else:
                if observation[2] == 1:
                    action = 0
                if observation[2] == 0:
                    action = self.nb_actions*1


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
