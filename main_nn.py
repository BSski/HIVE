import numpy as np
import gym

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from simagent.simultaneous import SimultaneousAgent

from npcs.alwayscooperate import AlwaysCooperateAgent
from npcs.alwaysdefect import AlwaysDefectAgent
from npcs.grim import GRIMAgent
from npcs.imperfecttft import ImperfectTitForTatAgent
from npcs.randomagent import RandomAgent
from npcs.suspicioustitfortat import SuspiciousTitForTatAgent
from npcs.titfortat import TitForTatAgent

from rl.agents import SARSAAgent, DDPGAgent
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from keras.models import model_from_json
import os


env = gym.make('gym_tdh:Tdh-v0')
np.random.seed(4125435)
env.seed(4125435)
nb_actions = 2
# print("Actions:", env.action_space.shape[0])
# print("States:", env.observation_space.shape[0])

WINDOW_LENGTH = 7

model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))
print(model.summary())

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn = DQNAgent(model, policy=policy, nb_actions=nb_actions, memory=memory)

model_one = clone_model(model)
policy_one = EpsGreedyQPolicy()
memory_one = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn_one = DQNAgent(model_one, policy=policy_one, nb_actions=nb_actions, memory=memory_one, enable_double_dqn=True, enable_dueling_network=True)

model_two = clone_model(model)
policy_two = EpsGreedyQPolicy()
memory_two = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn_two = DQNAgent(model_two, policy=policy_two, nb_actions=nb_actions, memory=memory_two, enable_double_dqn=True, enable_dueling_network=True)

model_three = clone_model(model)
policy_three = EpsGreedyQPolicy()
memory_three = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn_three = DQNAgent(model_three, policy=policy_three, nb_actions=nb_actions, memory=memory_three, enable_double_dqn=True, enable_dueling_network=True)

model_four = clone_model(model)
policy_four = EpsGreedyQPolicy()
memory_four = SequentialMemory(limit=3000, window_length=WINDOW_LENGTH)
dqn_four = DQNAgent(model_four, policy=policy_four, nb_actions=nb_actions, memory=memory_four, enable_double_dqn=True, enable_dueling_network=True)


alwayscoop = AlwaysCooperateAgent()
alwaysdefect = AlwaysDefectAgent()
grim = GRIMAgent()
imperfecttft = ImperfectTitForTatAgent()
randomagent = RandomAgent()
sus_titfortat= SuspiciousTitForTatAgent()
titfortat = TitForTatAgent()

sim_agent = SimultaneousAgent([dqn_one, dqn_two])#, dqn_two, dqn_three, dqn_four])
sim_agent.compile([Adam(lr=0.0001), Adam(lr=0.0001), Adam(lr=0.0001), Adam(lr=0.0001), Adam(lr=0.0001)], metrics=[['mae'], ['mae'], ['mae'], ['mae'], ['mae']])
his = sim_agent.fit(env, nb_steps=20000, verbose=2) # !! nb_steps % 200 == 0 !!
#sim_agent.test(env, nb_episodes=10, visualize=False)
# print(his.history)
print("\nDone!")


'''
# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# Load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''
