# HIVE
Requirements:
- Python 3.7.8 64 bit
- numpy==1.18.5
- pygame==1.9.6
- gym==0.17.2
- pandas==1.1.0
- matplotlib==3.3.0
- tensorflow==2.3.0
- keras_rl2==1.0.4
- Keras==2.4.3
- rl==3.0
- theano==1.0.5



:chart_with_upwards_trend: Changelog:

01.02.2021
- special no GUI version for conducting experiments

03.01.2021
- board's nature is hexagonal now instead of rectangular
- added optional controllable debug agent

23.12.2020
- the program is running in cycles now, X simulations in a row
- added appending data to .csv files after each simulation

10.12.2020
- using DQN instead of DDPG (DDPG couldn't handle TFT)
- UI changes

03.12.2020
- using DDPG instead of DQN now
- agents have parameters which impact their actions

21.11.2020
- agents now exist on 2d 43x43 plane and interact when on the same tile
- added visualisation done in pygame

04.11.2020
- more than two players can play now (interleaved training)
- added 7 NPCs with popular PD strategies
- players now include current episode step and type of an enemy in their observations

28.10.2020
- created stable, developable version of two neural nets playing PD against each other using:
  - custom OpenAI gym
  - custom keras-rl2 agent
  - custom keras-rl2 callbacks

21.09.2020
- created OpenAI custom gym environment
- connected it to keras-rl2, doesn't work yet, need to create a proper network for it

16.09.2020
- created the repository
- added prisoner's dilemma basic mechanism
