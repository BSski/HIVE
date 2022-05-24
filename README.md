<br>
<hr>
<b><p align="center">WARNING: VERY OLD CODE</p></b>
<i><p align="center">Most of the code was written in 2020. I have learned a lot since then and I am aware of the poor quality of the code.</p></i>
<hr>
<br>

<p align="center">
  <img src="./HIVE GUI.png" alt="HIVE GUI" width="95%" height="63%" align="center">
</p>


<p align="center">
  <h2 align="center">HIVE</h2>
</p>


<div markdown="1" align="center">

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/BSski/HIVE/blob/master/LICENSE)

</div>


## Table of contents
* [Project description](#scroll-project-description)
* [Technologies used](#hammer-technologies-used)
* [Room for improvement](#arrow_up-room-for-improvement)
* [Contact](#telephone_receiver-contact)
* [Changelog](#chart_with_upwards_trend-changelog)
* [Author](#construction_worker-author)
* [License](#unlock-license)


## :scroll: Project description
Platform created to facilitate conducting spatial multi-agent iterated prisoner's dilemma experiments between groups controlled by RL algorithm incorporating artificial neural networks. Uses Double Dueling Deep Q-Network, a deep reinforcement learning algorithm.

Based on <a href="https://github.com/wau/keras-rl2">keras-rl2</a>.
<br>
Simultaneous agent inspired by <a href="https://github.com/velochy/rl-bargaining/blob/master/interleaved.py">interleaved.py by Velochy</a>.</p>


## :hammer: Technologies used
- Python 3.7.11
- Pygame 1.9.6
- Numpy 1.18.5
- OpenAI gym 0.17.2
- Matplotlib 3.3.0
- Keras_rl2 1.0.4
- Tensorflow 2.3.0
- Theano 1.0.5


## :arrow_up: Room for improvement
This is an old project of mine and it would certainly benefit from:
- refactoring the code into better functions,
- refactoring the code into different files,
- getting rid of many anti-patterns,
- having tests written for it.


## :telephone_receiver: Contact
- <contact.bsski@gmail.com>


## :chart_with_upwards_trend: Changelog:

:date: 01.02.2021
- no GUI version for faster experiments

:date: 03.01.2021
- board's nature is hexagonal now instead of rectangular
- added optional controllable debug agent

:date: 23.12.2020
- the program is running in cycles now, X simulations in a row
- added appending data to .csv files after each simulation

:date: 10.12.2020
- using DQN instead of DDPG (DDPG couldn't handle TFT)
- UI changes

:date: 03.12.2020
- using DDPG instead of DQN now
- agents have parameters that impact their actions

:date: 21.11.2020
- agents now exist on 2d 43x43 plane and interact when on the same tile
- added visualisation done in pygame

:date: 04.11.2020
- more than two players can play now (interleaved training)
- added 7 NPCs with popular PD strategies
- players now include current episode step and type of the enemy in their observations

:date: 28.10.2020
- created stable, developable version of two neural nets playing PD against each other using:
  - custom OpenAI gym
  - custom keras-rl2 agent
  - custom keras-rl2 callbacks

:date: 21.09.2020
- created OpenAI custom gym environment
- connected it to keras-rl2, doesn't work yet, need to create a proper network for it

:date: 16.09.2020
- created the repository
- added prisoner's dilemma basic mechanism


## :construction_worker: Author
- [@BSski](https://www.github.com/BSski)


## :unlock: License
MIT
