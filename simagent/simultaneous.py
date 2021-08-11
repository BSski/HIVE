import numpy as np

from keras.layers import Concatenate
from os.path import splitext
from simagent.twoagentsagent import TwoAgentsAgent

class SimultaneousAgent(TwoAgentsAgent):
    def __init__(self, agents):
        self.agents = agents
        self.n = len(agents)
        self.compiled = False
        self.m_names = []
        self._training = False
        self._step = 0
        self.agents_combinations = []
        self.current_fight_number = -1
        self.current_fight = []
        self.player = 0
        self.player_one = 0
        self.game_step = 0
        super(SimultaneousAgent, self).__init__()

        self.current_game_number = -1  # Used to append steps to proper games.
        self.current_game = [], []
        self.agents_games_log = []

        # Creating list with proper structure.
        for i in range(self.n):
            self.agents_games_log.append([])

        # Creating list with proper structure.
        for i in range(self.n):
            case = []
            for j in range(self.n):
                case.append([])
            self.rewards_history.append(case)

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self,t):
        self._training = t
        for agent in self.agents:
            agent.training = t

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self,s):
        # print "setting step %i" % s
        self._step = s
        for agent in self.agents:
            agent.step = s

    def reset_states(self):
        for agent in self.agents:
            agent.reset_states()

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.
        # Argument
            observation (object): The current observation from the environment.
        # Returns
            The next action to be executed in the environment.
        """
        self.game_step = observation[2]
        if self.game_step == -1:
            if self.current_fight_number == len(self.agents_combinations)-1:
                self.current_fight_number = -1
            self.current_fight_number += 1

            # Print number of the pair playing right now / total number of pairs.
            ## print(self.current_fight_number, "/", len(self.agents_combinations)-1)
            self.current_fight = self.agents_combinations[self.current_fight_number]
            self.player = self.current_fight[0]
            self.player_one = self.current_fight[1]

            ## print("####### AGENTS PLAYING:",self.player, self.player_one)
            self.current_game[0].append([self.player, self.player_one])
            self.current_game[1].append([self.player_one, self.player])

            if self.current_fight_number == 0:
                self.current_game_number += 1

        ## print(observation[0], self.game_step, self.player_one, observation[3][1])
        ## print(observation[1], self.game_step, self.player, observation[3][0])

        current_players_actions = [self.agents[self.player].forward([observation[0],
                                                                     self.game_step,
                                                                     self.player_one,
                                                                     observation[3][1]
                                                                     ]),
                                   self.agents[self.player_one].forward([observation[1],
                                                                         self.game_step,
                                                                         self.player,
                                                                         observation[3][0]
                                                                         ])]

        if self.game_step != 19:
            self.current_game[0].append(current_players_actions[0])
            self.current_game[1].append(current_players_actions[1])

        return current_players_actions

    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.
        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        # Returns
            List of metrics values
        """
        if self.game_step == 19:
            self.agents_games_log[self.player].append([self.current_game[0],
                                                       self.current_game[1]])
            self.agents_games_log[self.player_one].append([self.current_game[1],
                                                           self.current_game[0]])
            self.current_game = [], []

        return [self.agents[self.player].backward(reward[0], terminal),
                self.agents[self.player_one].backward(reward[1], terminal)]

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.
        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        # Set optimizer and metrics (plus other 'compile' things) for each agent.
        for i,agent in enumerate(self.agents):
            if not agent.compiled:
                agent.compile(optimizer[i],metrics[i])

        # Create agents combinations.
        combinations = []
        for i in range(len(self.agents)):
            for j in range(len(self.agents)):
                if i != j:
                    combinations.append([i,j])

        for i in range(len(combinations)):
            combinations[i].sort()
        combinations.sort()

        for i in range(len(combinations)):
            if i % 2 == 0:
                self.agents_combinations.append(combinations[i])

        # Add each agent's metrics names to self.m_names.
        for i in range(len(self.agents)):
            self.m_names.append(self.agents[i].metrics_names)

        self.compiled = True

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.
        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        fbase, fext = splitext(filepath)
        for i, agent in enumerate(self.agents):
            if i <= 1:
                agent.load_weights('%s%i%s' % (fbase,i,fext))

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.
        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        fbase, fext = splitext(filepath)
        for i, agent in enumerate(self.agents):
            if i <= 1:
                agent.save_weights('%s%i%s' % (fbase,i,fext), overwrite)

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).
        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.
        # Returns
            A list of the model's layers
        """
        return [ layer for agent in self.agents
                    for layer in agent.layers() ]

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        # Returns
            A list of metric's names (string)
        """
        return [self.m_names[0], self.m_names[1]]

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        for agent in self.agents:
            agent._on_train_begin()

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        for agent in self.agents:
            agent._on_train_end()

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        for agent in self.agents:
            agent._on_test_begin()

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        for agent in self.agents:
            agent._on_test_end()



    def get_agents_games_log(self):
        return self.agents_games_log

    def get_n(self):
        return self.n

    def forward_test(self, observation, player0, player1, agent_index, agent_one_index):
        self.player = player0
        self.player_one = player1
        self.game_step = observation[2]

        if self.game_step == -1:
            ## print("####### HIVES PLAYING:", self.player, self.player_one)
            ## print("####### AGENTS PLAYING:", agent_index, agent_one_index)

            if self.current_fight_number == 0:
                self.current_game_number += 1
        ## print(observation)

        # [my_action, enemy_action, step, my_hive, enemy_hive, my_reward, my_index, enemy_index]
        current_players_actions = [self.agents[self.player].forward([self.game_step,
                                                                     observation[1],
                                                                     observation[0],
                                                                     self.player,
                                                                     self.player_one,
                                                                     observation[3][0],
                                                                     agent_index,
                                                                     agent_one_index
                                                                     ]),
                                   self.agents[self.player_one].forward([self.game_step,
                                                                         observation[0],
                                                                         observation[1],
                                                                         self.player_one,
                                                                         self.player,
                                                                         observation[3][1],
                                                                         agent_one_index,
                                                                         agent_index
                                                                         ])]

        return current_players_actions


    def backward_test(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.
        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        # Returns
            List of metrics values
        """
        if self.game_step == 19:
            self.agents_games_log[self.player].append([self.current_game[0],
                                                       self.current_game[1]])
            self.agents_games_log[self.player_one].append([self.current_game[1],
                                                           self.current_game[0]])
            self.current_game = [], []

        return [self.agents[self.player].backward(reward[0],terminal),
                self.agents[self.player_one].backward(reward[1],terminal)]

    def get_rewards_history(self):
        return self.rewards_history
