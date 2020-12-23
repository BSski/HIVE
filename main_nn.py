import matplotlib.pyplot as plt
import numpy as np
import pandas
import gym
import os
import sys


from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras.models import model_from_json
from simagent.simultaneous import SimultaneousAgent

from npcs.alwayscooperate import AlwaysCooperateAgent
from npcs.alwaysdefect import AlwaysDefectAgent
from npcs.grim import GRIMAgent
from npcs.imperfecttft import ImperfectTitForTatAgent
from npcs.randomagent import RandomAgent
from npcs.suspicioustitfortat import SuspiciousTitForTatAgent
from npcs.titfortat import TitForTatAgent

from rl.agents.ddpg import DDPGAgent
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

erase_csv_warning = 1

# Asking the user whether to overwrite all .csv files.
if erase_csv_warning == 1:
    answer = input("\nDo you want to overwrite all .csv files?: ")
    if answer != 'yes':
        sys.exit("\nYou decided not to proceed with the program.")


# Clearing all .csv files.
for i in range(9):
    for j in range(9):
        file_name = 'csv output/{}.csv'.format([i,j])
        # Opening the file with w+ mode truncates the file.
        f = open(file_name, "w+")
        f.close()


# Env initialization.
env = gym.make('gym_tdh:Tdh-v0')
np.random.seed(5000)
env.seed(5000)

# assert len(env.action_space.shape) == 1  # DDPG
# nb_actions = env.action_space.shape[0]  # DDPG
nb_actions = 2  # DQN
# print("Actions:", env.action_space.shape[0])
# print("States:", env.observation_space.shape[0])

# Hyperparameters settings.
WINDOW_LENGTH = 5
GAMMA = 2
GAMMA2 = 0.5

model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))
print(model.summary())

"""
actor = Sequential()
actor.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

actor_two = Sequential()
actor_two.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
actor_two.add(Dense(64))
actor_two.add(Activation('relu'))
actor_two.add(Dense(64))
actor_two.add(Activation('relu'))
actor_two.add(Dense(32))
actor_two.add(Activation('relu'))
actor_two.add(Dense(16))
actor_two.add(Activation('relu'))
actor_two.add(Dense(16))
actor_two.add(Activation('relu'))
actor_two.add(Dense(nb_actions))
actor_two.add(Activation('sigmoid'))


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(WINDOW_LENGTH,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('sigmoid')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

action_input_two = Input(shape=(nb_actions,), name='action_input_two')
observation_input_two = Input(shape=(WINDOW_LENGTH,) + env.observation_space.shape, name='observation_input_two')
flattened_observation_two = Flatten()(observation_input_two)
y = Concatenate()([action_input_two, flattened_observation_two])
y = Dense(64)(y)
y = Activation('relu')(y)
y = Dense(64)(y)
y = Activation('relu')(y)
y = Dense(32)(y)
y = Activation('relu')(y)
y = Dense(16)(y)
y = Activation('relu')(y)
y = Dense(16)(y)
y = Activation('relu')(y)
y = Dense(1)(y)
y = Activation('sigmoid')(y)
critic_two = Model(inputs=[action_input_two, observation_input_two], outputs=y)

# DDPG 1
memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agentddpg = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=GAMMA, target_model_update=1e-3)
"""

"""
# DDPG 2
memory_two = SequentialMemory(limit=100000, window_length=1)
random_process_two = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agentddpg_two = DDPGAgent(nb_actions=nb_actions, actor=actor_two, critic=critic_two, critic_action_input=action_input,
                  memory=memory_two, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process_two, gamma=.99, target_model_update=1e-3)
"""


policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn = DQNAgent(model, policy=policy, nb_actions=nb_actions, memory=memory, gamma = GAMMA)

model_one = clone_model(model)
policy_one = EpsGreedyQPolicy()
memory_one = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn_one = DQNAgent(model_one, policy=policy_one, nb_actions=nb_actions, memory=memory_one, gamma = GAMMA, enable_double_dqn=True, enable_dueling_network=True)

model_two = clone_model(model)
policy_two = EpsGreedyQPolicy()
memory_two = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn_two = DQNAgent(model_two, policy=policy_two, nb_actions=nb_actions, memory=memory_two, gamma = GAMMA, enable_double_dqn=True, enable_dueling_network=True)

model_three = clone_model(model)
policy_three = EpsGreedyQPolicy()
memory_three = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn_three = DQNAgent(model_three, policy=policy_three, nb_actions=nb_actions, memory=memory_three, gamma = GAMMA, enable_double_dqn=True, enable_dueling_network=True)

model_four = clone_model(model)
policy_four = EpsGreedyQPolicy()
memory_four = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn_four = DQNAgent(model_four, policy=policy_four, nb_actions=nb_actions, memory=memory_four, gamma = GAMMA2, enable_double_dqn=True, enable_dueling_network=True)

model_five = clone_model(model)
policy_five = EpsGreedyQPolicy()
memory_five = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn_five = DQNAgent(model_five, policy=policy_five, nb_actions=nb_actions, memory=memory_five, gamma = GAMMA2, enable_double_dqn=True, enable_dueling_network=True)

model_six = clone_model(model)
policy_six = EpsGreedyQPolicy()
memory_six = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn_six = DQNAgent(model_six, policy=policy_six, nb_actions=nb_actions, memory=memory_six, gamma = GAMMA2, enable_double_dqn=True, enable_dueling_network=True)


alwayscoop = AlwaysCooperateAgent()
alwaysdefect = AlwaysDefectAgent()
grim = GRIMAgent()
imperfecttft = ImperfectTitForTatAgent()
randomagent = RandomAgent()
sus_titfortat = SuspiciousTitForTatAgent()
titfortat = TitForTatAgent()


#############################################################################
############################      SETTINGS      #############################
#############################################################################

# Neural network agents first!
hives_list = [dqn_one, dqn_two, alwayscoop, alwaysdefect, grim, imperfecttft, randomagent, sus_titfortat, titfortat]
nb_agents_in_hives = [15,  # DQN_one
                      0,  # DQN_two
                      15,  # AlwaysCoop
                      0,  # AlwaysDefect
                      0,  # GRIM
                      0,  # Imperfect TFT
                      0,  # Random
                      0,  # Suspicious TFT
                      0  # Tit For Tat
                      ]

#############################################################################
#############################################################################
#############################################################################

optimizers_list = []
metrics_list = []
for i in range(len(hives_list)):
    optimizers_list.append(Adam(lr=0.001))
for i in range(len(hives_list)):
    metrics_list.append(['mae'])


# dodaj, ze jesli jeden i drugi agent sa NPCami, to nie odgrywa miedzy nimi gry
while True:
    sim_agent = SimultaneousAgent(hives_list)
    sim_agent.compile(optimizers_list,
                       metrics=metrics_list)
    sim_agent.save_weights('model.h5', overwrite=True)
    his = sim_agent.fit(env, verbose=2, nb_agents_in_hives=nb_agents_in_hives)
    sim_agent.load_weights('model.h5')
    # print(his.history)
    if sim_agent.exit_after_this_sim == 1:
        break


n = sim_agent.get_n()
agents_games_log = sim_agent.get_agents_games_log()
rewards_history = sim_agent.get_rewards_history()
