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

presentation_mode = 1

env = gym.make('gym_tdh:Tdh-v0')
np.random.seed()
env.seed()
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
model.add(Dense(16, activation = 'relu'))
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

alwayscoop = AlwaysCooperateAgent()
alwaysdefect = AlwaysDefectAgent()
grim = GRIMAgent()
imperfecttft = ImperfectTitForTatAgent()
randomagent = RandomAgent()
sus_titfortat = SuspiciousTitForTatAgent()
titfortat = TitForTatAgent()


if presentation_mode == 0:
    model.load_weights("model0.h5")
    model_one.load_weights("model1.h5")


# dodaj, ze jesli jeden i drugi agent sa NPCami, to nie odgrywa miedzy nimi gry
# Neural network agents first!
sim_agent = SimultaneousAgent([dqn_one, dqn_two, alwayscoop, alwaysdefect, grim, imperfecttft, randomagent, sus_titfortat, titfortat])
sim_agent.compile([Adam(lr=0.001), Adam(lr=0.001), Adam(lr=0.0001), Adam(lr=0.0001), Adam(lr=0.0001), Adam(lr=0.0001), Adam(lr=0.0001), Adam(lr=0.0001), Adam(lr=0.0001)],
                   metrics=[['mae'], ['mae'], ['mae'], ['mae'], ['mae'], ['mae'], ['mae'], ['mae'], ['mae']])
his = sim_agent.fit(env, nb_steps=720000, verbose=2) # !! assert nb_steps % 720 == 0 !!
#sim_agent.test(env, nb_episodes=30, visualize=False)
# print(his.history)
print("\nDone!")

fights_list = sim_agent.get_fights_list()
n = sim_agent.get_n()

#sim_agent.save_weights("model.h5")

agents_names = {
1: "DQN_one",
2: "DQN_two",
3: "AlwaysCoop",
4: "AlwaysDefect",
5: "GRIM",
6: "Imperfect TFT",
7: "Random",
8: "Suspicious TFT",
9: "Tit For Tat"
}

cls = lambda: os.system('cls')
cls()
percent_of_last_games = 0.05

while True:
    # Smaller number first.
    print("1: DQN_one\n2: DQN_two\n3: AlwaysCoop\n4: AlwaysDefect\n5: GRIM\n6: Imperfect TFT\n7: Random\n8: Suspicious TFT\n9: Tit For Tat")
    x = input("\nWhich agent do you want to see fight?: ")
    y = input("Who will be the opponent?: ")
    while x == '' or y == '' or int(x) > n or int(y) > n:
        cls()
        print("### Invalid input. ###\n")
        print("1: DQN_one\n2: DQN_two\n3: AlwaysCoop\n4: AlwaysDefect\n5: GRIM\n6: Imperfect TFT\n7: Random\n8: Suspicious TFT\n9: Tit For Tat")
        x = input("\nWhich agent do you want to see fight?: ")
        y = input("Who will be the opponent?: ")
    cls()
    x = int(x)-1
    y = int(y)-1
    if x > -1 and y > -1 and x < n and y < n:
        sim_agent.test_fight(env, nb_episodes=1, visualize=False, player=x, player_one=y)
        actions_history_list = []
        for i in range(int(len(fights_list[x])/(n-1)*percent_of_last_games)):
            #print(fights_list[x][int(len(fights_list[x])*0.5+ i*(n-1)+ (y-1))-4][0])
            for j in range(len(fights_list[x][int(len(fights_list[x])*percent_of_last_games+ i*(n-1)+ (y-1))-4][0])):
                if j == 0:
                    pass
                else:
                    actions_history_list.append(fights_list[x][int(len(fights_list[x])*percent_of_last_games+ i*(n-1)+ (y-1))-4][0][j])
        #print("\n", actions_history_list, "\n")

        print("\n\nMean actions of", agents_names[x+1], "against", agents_names[y+1], "in last", int(len(actions_history_list)/20), "games:")
        for j in range(20):
            actions_sum = 0
            for i in range(int(len(actions_history_list)/20)):
                actions_sum += actions_history_list[i*20+j]
            #(actions_sum)
            #print(int(len(actions_history_list)/20))
            print(round(actions_sum/int(len(actions_history_list)/20), 2), " ", end='')
        print("\n\n")


    else:
        print("### At least one of those agents doesn't exist or player number is bigger than opponent number. ###\n")



#zakres nagrod <-120:480>


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


"""
# player number must be smaller than opponent number
player = 3  # whose fights do you want to see?  between 1 and n
opponent = 5  # against which player?  between 1 and n
# iterate agent's fights
if player <= n and player > 0 and opponent <= n and opponent > 0 and player < opponent:
    for i in range(int(len(fights_list[player-1])/(n-1)*0.5)):
        #print(len(fights_list[player-1]))
        #print(int(len(fights_list[player-1])*0.5+ i*(n-1)+ opponent-1))
        print(fights_list[player-1][int(len(fights_list[player-1])*0.5+ i*(n-1)+ opponent-1)-1])
    print(int(len(fights_list)))
else:
    print("You chose a player that does not exist or player number is bigger or equal to opponent number.")
"""
