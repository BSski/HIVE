import numpy as np
import gym

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
#from keras.metrics import Recall
from simultaneous import SimultaneousAgent

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

#
from keras.models import model_from_json
import os
#

####
env = gym.make('gym_tdh:Tdh-v0')  # gym_tdh:Tdh-v0
np.random.seed()
env.seed()
nb_actions = 2
# print("Actions:", env.action_space.shape[0])
# states = env.observation_space.shape[0]
# print("States:", states)
WINDOW_LENGTH = 19

model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))
print(model.summary())


policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=6000, window_length=WINDOW_LENGTH)
dqn = DQNAgent(model, policy=policy, nb_actions=nb_actions, memory=memory)

model_one = clone_model(model)
policy_one = EpsGreedyQPolicy()
memory_one = SequentialMemory(limit=6000, window_length=WINDOW_LENGTH)
dqn_one = DQNAgent(model_one, policy=policy_one, nb_actions=nb_actions, memory=memory_one)


sim_agent = SimultaneousAgent([dqn, dqn])


sim_agent.compile([Adam(lr=0.001), Adam(lr=0.001)], metrics=[['mae'], ['mae']])

his = sim_agent.fit(env, nb_steps=30000, verbose=2)
sim_agent.test(env, nb_episodes=10, visualize=False)
# print(his.history)
print("Done!")
# sim_agent.test(env, nb_episodes=300, visualize=False)

'''
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

'''
'''
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=8000, window_length=WINDOW_LENGTH)
dqn = DQNAgent(loaded_model, policy=policy, nb_actions=nb_actions, memory=memory)
dqn.compile(Adam(lr=0.0001), metrics=['mse'])

dqn.fit(env, nb_steps=12000, verbose=2)

mem = ( self.memory, self.memory.actions,
self.memory.rewards,
self.memory.terminals,
self.memory.observations )
cPickle.dump(mem, open(self.memoryfile, "wb"), protocol=-1) # highest protocol means binary format

(self.memory, self.memory.actions,
self.memory.rewards,
self.memory.terminals,
self.memory.observations) = cPickle.load( open(self.memoryfile, "rb"))
'''
