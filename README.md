# HIVE
Requirements:
- numpy==1.18.5
- Keras==2.4.3
- keras_rl2==1.0.4
- tensorflow==2.3.0
- gym==0.17.2
- rl==3.0
- theano==1.0.5

Changelog:

04.11
- more than two players can play now (interleaved training)
- added 7 NPCs with popular PD strategies
- players now include current episode step and type of an enemy in their observations

28.10
- created stable, developable version of two neural nets playing PD against each other using:
  - custom OpenAI gym
  - custom keras-rl2 agent
  - custom keras-rl2 callbacks

21.09
- created OpenAI custom gym environment
- connected it to keras-rl2, doesn't work yet, need to create a proper network for it

16.09
- created the repository
- added prisoner's dilemma basic mechanism
