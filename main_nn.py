import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


####
env = gym.make('gym_tdh:Tdh-v0')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


model = Sequential()
model.add(Dense(2, activation='relu', input_dim = 1))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model, policy=None, nb_actions = nb_actions, memory = memory)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

####
dqn.fit(env, nb_steps=1000, verbose=1)


dqn.test(env, nb_episodes=5, visualize=True)
