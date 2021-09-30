import numpy as np
import gym
import sys
import tensorflow

from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from simagent.simultaneous import SimultaneousAgent

from npcs.alwayscooperate import AlwaysCooperateAgent
from npcs.alwaysdefect import AlwaysDefectAgent
from npcs.grim import GRIMAgent
from npcs.imperfecttft import ImperfectTitForTatAgent
from npcs.randomagent import RandomAgent
from npcs.suspicioustitfortat import SuspiciousTitForTatAgent
from npcs.titfortat import TitForTatAgent

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

#############################################################################
# CSV part
# Warning before overwriting data flag.
erase_csv_warning = 1

# Asking the user whether to overwrite all .csv files.
if erase_csv_warning == 1:
    warning = "\n\n*** WARNING: Running this file will overwrite all previous.csv files. Do you want to proceed? [yes/no]: *** "
    answer = input(warning)

    if answer != 'yes' and answer != 'Yes' and answer != 'YES':
        sys.exit("\n*** You decided not to proceed with the program. ***")

# TODO: Change to 'with' so it's safe.
# Clearing all .csv files.
for i in range(9):
    for j in range(9):
        file_name = 'csv output/{}.csv'.format([i, j])
        f = open(file_name, "w+")
        f.write("")
        f.close()

#############################################################################

# Env initialization.
env = gym.make('gym_tdh:Tdh-v0')
np.random.seed()
env.seed()

nb_actions = 10
# print("Actions:", env.action_space.shape[0])
# print("States:", env.observation_space.shape[0])

# Hyperparameters settings.
WINDOW_LENGTH = 4
GAMMA = 0.90
nb_steps_warmup = 200

model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_actions, activation='sigmoid'))
print(model.summary())


policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=30000, window_length=WINDOW_LENGTH)
dqn = DQNAgent(model, policy=policy,
               nb_actions=nb_actions, memory=memory, gamma=GAMMA)

model_one = clone_model(model)
policy_one = EpsGreedyQPolicy()
memory_one = SequentialMemory(limit=30000, window_length=WINDOW_LENGTH)
dqn_one = DQNAgent(model_one, policy=policy_one,
                   nb_actions=nb_actions, memory=memory_one,
                   gamma=GAMMA, nb_steps_warmup=nb_steps_warmup,
                   enable_double_dqn=True, enable_dueling_network=True)

model_two = clone_model(model)
policy_two = EpsGreedyQPolicy()
memory_two = SequentialMemory(limit=30000, window_length=WINDOW_LENGTH)
dqn_two = DQNAgent(model_two, policy=policy_two,
                   nb_actions=nb_actions, memory=memory_two,
                   gamma=GAMMA, nb_steps_warmup=nb_steps_warmup,
                   enable_double_dqn=True, enable_dueling_network=True)

model_three = clone_model(model)
policy_three = EpsGreedyQPolicy()
memory_three = SequentialMemory(limit=30000, window_length=WINDOW_LENGTH)
dqn_three = DQNAgent(model_three, policy=policy_three,
                     nb_actions=nb_actions, memory=memory_three,
                     gamma=GAMMA, nb_steps_warmup=nb_steps_warmup,
                     enable_double_dqn=True, enable_dueling_network=True)

model_four = clone_model(model)
policy_four = EpsGreedyQPolicy()
memory_four = SequentialMemory(limit=30000, window_length=WINDOW_LENGTH)
dqn_four = DQNAgent(model_four, policy=policy_four,
                    nb_actions=nb_actions, memory=memory_four,
                    gamma=GAMMA, nb_steps_warmup=nb_steps_warmup,
                    enable_double_dqn=True, enable_dueling_network=True)

model_five = clone_model(model)
policy_five = EpsGreedyQPolicy()
memory_five = SequentialMemory(limit=30000, window_length=WINDOW_LENGTH)
dqn_five = DQNAgent(model_five, policy=policy_five,
                    nb_actions=nb_actions, memory=memory_five,
                    gamma=GAMMA, nb_steps_warmup=nb_steps_warmup,
                    enable_double_dqn=True, enable_dueling_network=True)

model_six = clone_model(model)
policy_six = EpsGreedyQPolicy()
memory_six = SequentialMemory(limit=30000, window_length=WINDOW_LENGTH)
dqn_six = DQNAgent(model_six, policy=policy_six,
                   nb_actions=nb_actions, memory=memory_six,
                   gamma=GAMMA, nb_steps_warmup=nb_steps_warmup,
                   enable_double_dqn=True, enable_dueling_network=True)


alwayscoop = AlwaysCooperateAgent()
alwaysdefect = AlwaysDefectAgent(nb_actions)
grim = GRIMAgent(nb_actions)
imperfecttft = ImperfectTitForTatAgent(nb_actions)
randomagent = RandomAgent(nb_actions)
sus_titfortat = SuspiciousTitForTatAgent(nb_actions)
titfortat = TitForTatAgent(nb_actions)


#############################################################################
############################      SETTINGS      #############################
#############################################################################

# Neural network agents first!
hives_list = [dqn_one, dqn_two, dqn_three, dqn_four,
              dqn_five, dqn_six, randomagent, sus_titfortat, titfortat]
nb_agents_in_hives = [10,  # DQN_one
                      10,  # DQN_two
                      10,  # AlwaysCoop
                      10,  # AlwaysDefect
                      10,  # GRIM
                      10,  # Imperfect TFT
                      0,   # Random
                      0,   # Suspicious TFT
                      0    # Tit For Tat
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

# Conduct the experiment.
for i in range(30):
    sim_agent = SimultaneousAgent(hives_list)
    sim_agent.compile(optimizers_list, metrics=metrics_list)
    sim_agent.save_weights('model.h5', overwrite=True)
    sim_agent.fit(env, verbose=2,
                  nb_agents_in_hives=nb_agents_in_hives)  # , debug_agent = 1)
    sim_agent.load_weights('model.h5')
    if sim_agent.exit_after_this_sim == 1:
        break

print("\n\n\n*** Done! The results are ready. ***\n\n")
