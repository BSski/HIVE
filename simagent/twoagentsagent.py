# -*- coding: utf-8 -*-
import warnings
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as pp
from keras.callbacks import History

from callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)

import pygame
import random
from grid import grid
from grid import grid_hex
from positions import *
from positions import boundary_tiles
from csv import writer
import time


class TwoAgentsAgent(object):
    """Abstract base class for all implemented agents.
    Each agent interacts with the environment (as defined by the `Env` class) by first observing the
    state of the environment. Based on this observation the agent changes the environment by performing
    an action.
    Do not use this abstract base class directly but instead use one of the concrete agents implemented.
    Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
    interface, you can use them interchangeably.
    To implement your own agent, you have to implement the following methods:
    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`
    - `layers`
    # Arguments
        processor (`Processor` instance): See [Processor](#processor) for details.
    """

    def __init__(self, processor=None):
        self.processor = processor
        self.training = False
        self.step = 0
        self.rewards_history = []
        self.exit_after_this_sim = 0

    def get_config(self):
        """Configuration of the agent for serialization.
        # Returns
            Dictionnary with agent configuration
        """
        return {}

    def fit(self, env, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None, nb_agents_in_hives=[0,0,0,0,0,0,0,0,0], debug_agent = 0):
        """Trains the agent on the given environment.
        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': 1,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        ##########################################################################
        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False

        ##########################################################################
        def fight(self, env, action_repetition, callbacks, nb_max_start_steps, start_step_policy,
                  nb_max_episode_steps, episode_reward, episode_step, observation, episode, did_abort,
                  player = None, player_one = None, agent_index = None, agent_one_index = None,
                  parameters = [], parameters_one = []):
            try:
                print(player, player_one, parameters, parameters_one)
                for i_step in range(20):
                    if observation is None:  # start of a new episode
                        callbacks.on_episode_begin(episode)
                        episode_step = np.int16(0)
                        episode_reward = [0, 0]

                        # Obtain the initial observation by resetting the environment.
                        self.reset_states()
                        observation = deepcopy(env.reset())
                        # if self.processor is not None:
                            # observation = self.processor.process_observation(
                                # observation)
                        assert observation is not None

                    # nb_random_start_steps loop was there, but it's gone now, since I'm doing that in env #

                    # At this point, we expect to be fully initialized.
                    assert episode_reward is not None
                    assert episode_step is not None
                    assert observation is not None

                    # Run a single step.
                    callbacks.on_step_begin(episode_step)
                    # This is where all of the work happens. We first perceive and compute the action
                    # (forward step) and then use the reward to improve (backward step).
                    #action = self.forward_test(observation,
                                              # player,
                                              # player_one,
                                              # agent_index,
                                              # agent_one_index
                                              # )
                    # if self.processor is not None:
                        # action = self.processor.process_action(action)
                    action = [1,1]
                    # Parameters impact.
                    print("actions before parameters:", action)
                    #action[0] = (action[0] + (0.05 * parameters[0]))*(1-(parameters[1]*0.05))
                    #action[1] = (action[1] + (0.05 * parameters_one[0]))*(1-(parameters_one[1]*0.05))


                    reward = [0, 0]
                    accumulated_info = {}
                    done = False
                    for _ in range(action_repetition):
                        callbacks.on_action_begin(action)
                        observation, r, done, info = env.step(action)
                        observation = deepcopy(observation)
                        # if self.processor is not None:
                            # observation, r, done, info = self.processor.process_step(
                                # observation, r, done, info)
                        for key, value in info.items():
                            if not np.isreal(value):
                                continue
                            if key not in accumulated_info:
                                accumulated_info[key] = np.zeros_like(value)
                            accumulated_info[key] += value
                        callbacks.on_action_end(action)
                        reward[0] += r[0]
                        reward[1] += r[1]

                        if done:
                            break

                    # nb_max_episode_steps was there, but it's gone now, since I'm doing that in env #

                    #metrics, metrics1 = self.backward_test(reward, terminal=done)
                    metrics = 0,0,0
                    metrics1 = 0,0,0
                    episode_reward[0] += reward[0]
                    episode_reward[1] += reward[1]
                    print("### episode_reward", episode_reward)
                    step_logs = {
                        'action': action[0],
                        'action1': action[1],
                        'observation': observation[0],
                        'observation1': observation[1],
                        'reward': reward[0],
                        'reward1': reward[1],
                        'metrics': metrics,
                        'metrics1': metrics1,
                        'episode': episode,
                        'info': accumulated_info,
                    }
                    callbacks.on_step_end(episode_step, step_logs)
                    episode_step += 1
                    self.step += 1
                    if done:
                        # We are in a terminal state but the agent hasn't yet seen it. We therefore
                        # perform one more forward-backward call and simply ignore the action before
                        # resetting the environment. We need to pass in `terminal=False` here since
                        # the *next* state, that is the state of the newly reset environment, is
                        # always non-terminal by convention.
                        #self.forward_test(observation,
                                          #player,
                                          ##player_one,
                                         # agent_index,
                                         # agent_one_index
                                         # )
                        #self.backward_test([0, 0], terminal=False)

                        # This episode is finished, report and reset.
                        episode_logs = {
                            'episode_reward': episode_reward,
                            'nb_episode_steps': episode_step,
                            'nb_steps': self.step,
                        }
                        callbacks.on_episode_end(episode, episode_logs)

                        episode += 1
                        observation = None
                        episode_step = None
            except KeyboardInterrupt:
                # We catch keyboard interrupts here so that training can be be safely aborted.
                # This is so common that we've built this right into this function, which ensures that
                # the `on_train_end` method is properly called.
                did_abort = True

            return episode_reward, episode_step, observation, episode, did_abort

##########################################################################
##########################################################################
##########################################################################

        pygame.init()
        size = (860, 670)
        pygame.display.set_caption("HIVE")
        screen = pygame.display.set_mode(size)
        hive_icon = pygame.image.load('sprites/hive.png')
        pygame.display.set_icon(hive_icon)
        # Set the main loop to not done and initialize a clock.
        pygame_done = False
        clock = pygame.time.Clock()

        font  = pygame.font.SysFont("liberationmono", 13)
        font2 = pygame.font.SysFont("liberationmono", 12)
        font3 = pygame.font.SysFont("liberationmono", 11)
        font4 = pygame.font.SysFont("humorsans", 70)
        font5 = pygame.font.SysFont("liberationmono", 12)
        font6 = pygame.font.SysFont("liberationmono", 14)
        font7 = pygame.font.SysFont("liberationmono", 18)

        # Defining colors.
        BLACK       = (   0,   0,   0)
        WHITE       = ( 255, 255, 255)
        GRAY        = ( 210, 210, 210)
        LIGHTGRAY   = ( 239, 239, 239)
        MIDDLEGRAY  = ( 180, 180, 180)
        DARKGRAY    = ( 130, 130, 130)
        DARKERGRAY  = (  45,  45,  45)
        FORESTGREEN = (  34, 139,  34)
        DARKRED     = ( 178,  34,  34)
        DARKBLUE    = (   0,   0, 139)
        GREEN       = (  0,  230, 115)
        BLUE        = (   0, 102, 204)
        RED         = ( 255,  77,  77)

        # Load images.
        PLUS_UP = pygame.image.load("sprites/plus_up.png")
        PLUS_DOWN = pygame.image.load("sprites/plus_down.png")
        MINUS_UP = pygame.image.load("sprites/minus_up.png")
        MINUS_DOWN = pygame.image.load("sprites/minus_up.png")
        RESET_UP = pygame.image.load("sprites/reset_up.png")
        RESET_DOWN = pygame.image.load("sprites/reset_down.png")
        START_UP = pygame.image.load("sprites/start_up.png")
        START_DOWN = pygame.image.load("sprites/start_down.png")
        PAUSE_UP = pygame.image.load("sprites/pause_up.png")
        PAUSE_DOWN = pygame.image.load("sprites/pause_down.png")

        animation = "|/-\\"

        carnivores = []
        btn_tempo_plus_clicked = 0
        btn_tempo_minus_clicked = 0

        tempo = 1                  # between 0.01 and 1    # default = 0.28
        counter = 0
        counter_prev = counter
        counter_for_fps = 0
        total_cycles_counter = 0

        big_counter = 0
        big_counter_prev = big_counter
        pause = 0
        chosen_cycles_per_second = 0
        cycles_per_sec_list = [30, 60, 90, 120, 150, 180, 240, 300, 360, 450, 600, 720, 900, 1200]
        cycles_per_sec_dividers_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40]
        fights_queue = []
        agents_fighting_queue = set()
        grid_size = 45  # max amount of agents = (grid_size-2)^2, if you exceed, they won't have space to spawn, max 43
        fight_flag = 0
        sum_of_cash_for_hives = []
        amount_of_games_to_log = 20
        debug_controllable_agent_present = debug_agent

        hives_balance_against = [
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        ]

        def draw_window():
            logo = font4.render("H I V E", True, (80, 80, 80))
            screen.blit(logo, (28, 18))
            signature = font3.render("bsski 2020", True, (200, 200, 200))
            screen.blit(signature, (70, 60))
            """
            # Drawing grid.
            square = 10
            square_width = square
            square_height = square
            y = -square_height + 10
            for i in range(0, grid_size):
                y += square_height
                x = -square_width + 213
                for j in range(0, grid_size):
                    x += square_width
                    pygame.draw.rect(screen, GRAY, [x, y, square_width-1, square_height-1])
            """
            # Charts backgrounds.
            # pygame.draw.rect(screen, GRAY, [860, 44, 176, 6])

            # Generate hex map.
            x = 213
            y = 9
            diameter = 45
            for j in range((int(diameter/2)+1),(diameter*2)-1):
                if j > diameter:
                    if diameter - (j-diameter) < int(diameter/2)+1:
                        break
                    else:
                        row_quantity = diameter - (j-diameter)
                else:
                    row_quantity = j
                for i in range(0, row_quantity):
                    shift = int((diameter - row_quantity)/2)-1
                    if j % 2 == 0:
                        pygame.draw.line(screen, GRAY, (x+2+(10*i)+5+(shift*10), y+((j-int(diameter/2))*9)-1), (x+4+2+(10*i)+5+(shift*10), y+((j-int(diameter/2))*9)-1), 1)
                        pygame.draw.line(screen, GRAY, (x+1+(10*i)+5+(shift*10), y+((j-int(diameter/2))*9)-1+1), (x+6+1+(10*i)+5+(shift*10), y+((j-int(diameter/2))*9)-1+1), 1)
                        pygame.draw.rect(screen, GRAY, [x+(10*i)+5+(shift*10), y+((j-int(diameter/2))*9)+1, 9, 5])
                        pygame.draw.line(screen, GRAY, (x+1+(10*i)+5+(shift*10), y+((j-int(diameter/2))*9)-1+7), (x+6+1+(10*i)+5+(shift*10), y+((j-int(diameter/2))*9)-1+7), 1)
                        pygame.draw.line(screen, GRAY, (x+2+(10*i)+5+(shift*10), y+((j-int(diameter/2))*9)-1+8), (x+4+2+(10*i)+5+(shift*10), y+((j-int(diameter/2))*9)-1+8), 1)
                    else:
                        pygame.draw.line(screen, GRAY, (x+2+(10*i)+(shift*10), y+((j-int(diameter/2))*9)-1), (x+4+2+(10*i)+(shift*10), y+((j-int(diameter/2))*9)-1), 1)
                        pygame.draw.line(screen, GRAY, (x+1+(10*i)+(shift*10), y+((j-int(diameter/2))*9)-1+1), (x+6+1+(10*i)+(shift*10), y+((j-int(diameter/2))*9)-1+1), 1)
                        pygame.draw.rect(screen, GRAY, [x+(10*i)+(shift*10), y+((j-int(diameter/2))*9)+1, 9, 5])
                        pygame.draw.line(screen, GRAY, (x+1+(10*i)+(shift*10), y+((j-int(diameter/2))*9)-1+7), (x+6+1+(10*i)+(shift*10), y+((j-int(diameter/2))*9)-1+7), 1)
                        pygame.draw.line(screen, GRAY, (x+2+(10*i)+(shift*10), y+((j-int(diameter/2))*9)-1+8), (x+4+2+(10*i)+(shift*10), y+((j-int(diameter/2))*9)-1+8), 1)

            ###

            for i in boundary_tiles:
                pygame.draw.line(screen, DARKGRAY, (grid_hex[i[1]][i[0]][0]+2, grid_hex[i[1]][i[0]][1]), (grid_hex[i[1]][i[0]][0]+2+4, grid_hex[i[1]][i[0]][1]), 1)
                pygame.draw.line(screen, DARKGRAY, (grid_hex[i[1]][i[0]][0]+1, grid_hex[i[1]][i[0]][1]+1), (grid_hex[i[1]][i[0]][0]+1+6, grid_hex[i[1]][i[0]][1]+1), 1)
                pygame.draw.rect(screen, DARKGRAY, [grid_hex[i[1]][i[0]][0], grid_hex[i[1]][i[0]][1]+2, 9, 5])
                pygame.draw.line(screen, DARKGRAY, (grid_hex[i[1]][i[0]][0]+1, grid_hex[i[1]][i[0]][1]+7), (grid_hex[i[1]][i[0]][0]+1+6, grid_hex[i[1]][i[0]][1]+7), 1)
                pygame.draw.line(screen, DARKGRAY, (grid_hex[i[1]][i[0]][0]+2, grid_hex[i[1]][i[0]][1]+8), (grid_hex[i[1]][i[0]][0]+2+4, grid_hex[i[1]][i[0]][1]+8), 1)

            ###

            # Main interface lines.
            pygame.draw.line(screen, GRAY, (12, 12), (12, 657), 1)
            pygame.draw.line(screen, GRAY, (198, 12), (198, 436), 1)
            pygame.draw.line(screen, GRAY, (656, 12), (656, 436), 1)
            pygame.draw.line(screen, GRAY, (847, 12), (847, 657), 1)

            icons_div = 88
            # Hive 1 icon.
            pygame.draw.rect(screen, ( 255, 255, 255), [25 , (icons_div+15*0), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*0)-1, 11, 11], 1)
            text_to_blit = font2.render(str(nb_agents_in_hives[0]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*0))
            # Hive 2 icon.
            pygame.draw.rect(screen, ( 255,   0,   0), [25, (icons_div+15*1), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*1)-1, 11, 11], 1)
            text_to_blit = font2.render(str(nb_agents_in_hives[1]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*1))
            # Hive 3 icon.
            pygame.draw.rect(screen, (   0, 255,   0), [25, (icons_div+15*2), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*2)-1, 11, 11], 1)
            text_to_blit = font2.render(str(nb_agents_in_hives[2]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*2))
            # Hive 4 icon.
            pygame.draw.rect(screen, (   0,   0, 255), [25, (icons_div+15*3), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*3)-1, 11, 11], 1)
            text_to_blit = font2.render(str(nb_agents_in_hives[3]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*3))
            # Hive 5 icon.
            pygame.draw.rect(screen, ( 255, 255,   0), [25, (icons_div+15*4), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*4)-1, 11, 11], 1)
            text_to_blit = font2.render(str(nb_agents_in_hives[4]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*4))
            # Hive 6 icon.
            pygame.draw.rect(screen, (   0, 255, 255), [25, (icons_div+15*5), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*5)-1, 11, 11], 1)
            text_to_blit = font2.render(str(nb_agents_in_hives[5]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*5))
            # Hive 7 icon.
            pygame.draw.rect(screen, ( 255,   0, 255), [25, (icons_div+15*6), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*6)-1, 11, 11], 1)
            text_to_blit = font2.render(str(nb_agents_in_hives[6]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*6))
            # Hive 8 icon.
            pygame.draw.rect(screen, ( 128, 128,   0), [25, (icons_div+15*7), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*7)-1, 11, 11], 1)
            text_to_blit = font2.render(str(nb_agents_in_hives[7]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*7))
            # Hive 9 icon.
            pygame.draw.rect(screen, (   0, 128,   0), [25, (icons_div+15*8), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*8)-1, 11, 11], 1)
            text_to_blit = font2.render(str(nb_agents_in_hives[8]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*8))



            if counter_for_fps % cycles_per_sec_dividers_list[chosen_cycles_per_second] == 0:
                temp_sum_of_cash_for_hives = []
                for i in range(len(hives_balance_against)):
                    temp_sum = 0
                    for j in range(len(hives_balance_against[i])):
                        temp_sum += hives_balance_against[i][j]
                    temp_sum_of_cash_for_hives.append(temp_sum)
                #print("SUM:", sum_of_cash_for_hives)
                sum_of_cash_for_hives = temp_sum_of_cash_for_hives


            icons_div = 88
            # Hive 1 icon.
            pygame.draw.rect(screen, ( 255, 255, 255), [25 , (icons_div+15*10), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*10)-1, 11, 11], 1)
            text_to_blit = font2.render(str(sum_of_cash_for_hives[0]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*10))
            # Hive 2 icon.
            pygame.draw.rect(screen, ( 255,   0,   0), [25, (icons_div+15*11), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*11)-1, 11, 11], 1)
            text_to_blit = font2.render(str(sum_of_cash_for_hives[1]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*11))
            # Hive 3 icon.
            pygame.draw.rect(screen, (   0, 255,   0), [25, (icons_div+15*12), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*12)-1, 11, 11], 1)
            text_to_blit = font2.render(str(sum_of_cash_for_hives[2]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*12))
            # Hive 4 icon.
            pygame.draw.rect(screen, (   0,   0, 255), [25, (icons_div+15*13), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*13)-1, 11, 11], 1)
            text_to_blit = font2.render(str(sum_of_cash_for_hives[3]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*13))
            # Hive 5 icon.
            pygame.draw.rect(screen, ( 255, 255,   0), [25, (icons_div+15*14), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*14)-1, 11, 11], 1)
            text_to_blit = font2.render(str(sum_of_cash_for_hives[4]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*14))
            # Hive 6 icon.
            pygame.draw.rect(screen, (   0, 255, 255), [25, (icons_div+15*15), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*15)-1, 11, 11], 1)
            text_to_blit = font2.render(str(sum_of_cash_for_hives[5]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*15))
            # Hive 7 icon.
            pygame.draw.rect(screen, ( 255,   0, 255), [25, (icons_div+15*16), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*16)-1, 11, 11], 1)
            text_to_blit = font2.render(str(sum_of_cash_for_hives[6]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*16))
            # Hive 8 icon.
            pygame.draw.rect(screen, ( 128, 128,   0), [25, (icons_div+15*17), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*17)-1, 11, 11], 1)
            text_to_blit = font2.render(str(sum_of_cash_for_hives[7]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*17))
            # Hive 9 icon.
            pygame.draw.rect(screen, (   0, 128,   0), [25, (icons_div+15*18), 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [25-1, (icons_div+15*18)-1, 11, 11], 1)
            text_to_blit = font2.render(str(sum_of_cash_for_hives[8]), True, (50, 50, 50))
            screen.blit(text_to_blit, (38, icons_div+15*18))


            table_interline = 18
            table_columns_width = 65
            table_starting_y = 460
            table_starting_x = 200

            pygame.draw.line(screen, DARKERGRAY, (table_starting_x, 460+table_interline*1.8), (table_starting_x+table_columns_width*7, 460+table_interline*1.8), 1)
            pygame.draw.line(screen, DARKERGRAY, (table_starting_x+table_columns_width*0.5, 460+table_interline*1.2), (table_starting_x+table_columns_width*0.5, 460+table_interline*8), 1)

            text_to_blit = font2.render("H1", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x+7, table_starting_y +table_interline*2))
            text_to_blit = font2.render("H2", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x+7, table_starting_y+table_interline*3))
            text_to_blit = font2.render("H3", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x+7, table_starting_y+table_interline*4))
            text_to_blit = font2.render("H4", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x+7, table_starting_y+table_interline*5))
            text_to_blit = font2.render("H5", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x+7, table_starting_y+table_interline*6))
            text_to_blit = font2.render("H6", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x+7, table_starting_y+table_interline*7))

            table_starting_x = table_starting_x+table_columns_width
            text_to_blit = font2.render("H1", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*1))
            text_to_blit = font2.render("-", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*2))
            text_to_blit = font2.render(str(hives_balance_against[0][1]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*3))
            text_to_blit = font2.render(str(hives_balance_against[0][2]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*4))
            text_to_blit = font2.render(str(hives_balance_against[0][3]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*5))
            text_to_blit = font2.render(str(hives_balance_against[0][4]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*6))
            text_to_blit = font2.render(str(hives_balance_against[0][5]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*7))

            table_starting_x = table_starting_x+table_columns_width
            text_to_blit = font2.render("H2", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*1))
            text_to_blit = font2.render(str(hives_balance_against[1][0]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*2))
            text_to_blit = font2.render("-", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*3))
            text_to_blit = font2.render(str(hives_balance_against[1][2]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*4))
            text_to_blit = font2.render(str(hives_balance_against[1][3]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*5))
            text_to_blit = font2.render(str(hives_balance_against[1][4]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*6))
            text_to_blit = font2.render(str(hives_balance_against[1][5]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*7))

            table_starting_x = table_starting_x+table_columns_width
            text_to_blit = font2.render("H3", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*1))
            text_to_blit = font2.render(str(hives_balance_against[2][0]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*2))
            text_to_blit = font2.render(str(hives_balance_against[2][1]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*3))
            text_to_blit = font2.render("-", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*4))
            text_to_blit = font2.render(str(hives_balance_against[2][3]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*5))
            text_to_blit = font2.render(str(hives_balance_against[2][4]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*6))
            text_to_blit = font2.render(str(hives_balance_against[2][5]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*7))

            table_starting_x = table_starting_x+table_columns_width
            text_to_blit = font2.render("H4", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*1))
            text_to_blit = font2.render(str(hives_balance_against[3][0]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*2))
            text_to_blit = font2.render(str(hives_balance_against[3][1]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*3))
            text_to_blit = font2.render(str(hives_balance_against[3][2]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*4))
            text_to_blit = font2.render("-", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*5))
            text_to_blit = font2.render(str(hives_balance_against[3][4]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*6))
            text_to_blit = font2.render(str(hives_balance_against[3][5]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*7))

            table_starting_x = table_starting_x+table_columns_width
            text_to_blit = font2.render("H5", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*1))
            text_to_blit = font2.render(str(hives_balance_against[4][0]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*2))
            text_to_blit = font2.render(str(hives_balance_against[4][1]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*3))
            text_to_blit = font2.render(str(hives_balance_against[4][2]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*4))
            text_to_blit = font2.render(str(hives_balance_against[4][3]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*5))
            text_to_blit = font2.render("-", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*6))
            text_to_blit = font2.render(str(hives_balance_against[4][5]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*7))

            table_starting_x = table_starting_x+table_columns_width
            text_to_blit = font2.render("H6", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*1))
            text_to_blit = font2.render(str(hives_balance_against[5][0]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*2))
            text_to_blit = font2.render(str(hives_balance_against[5][1]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*3))
            text_to_blit = font2.render(str(hives_balance_against[5][2]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*4))
            text_to_blit = font2.render(str(hives_balance_against[5][3]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*5))
            text_to_blit = font2.render(str(hives_balance_against[5][4]), True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*6))
            text_to_blit = font2.render("-", True, (50, 50, 50))
            screen.blit(text_to_blit, (table_starting_x, table_starting_y+table_interline*7))

            text_to_blit = font2.render("exit after this sim.? (backspace): " + str(self.exit_after_this_sim), True, (50, 50, 50))
            screen.blit(text_to_blit, (19, 645))


        # Class creating animals.
        class Animal:
            def __init__(self, coord_x, coord_y, index, dna):
                pass

            def get_index(self):
                return self.index

            def get_dna(self):
                return self.dna

            def get_state(self):
                if self.energy > 0: return 1 #alive
                if self.energy < 1: return 0 #dead

            def get_energy(self):
                return self.energy

            def set_energy(self, new_energy):
                self.energy = new_energy

            def get_coords(self):
                return (self.coord_x, self.coord_y)

        # Class creating carnivores.
        class Carnivore(Animal):
            def __init__(self, coord_x, coord_y, index, hive, parameters):
                self.coord_x = coord_x
                self.coord_y = coord_y
                self.previous_coord_x = self.coord_x
                self.previous_coord_y = self.coord_y
                self.account = 5000
                self.index = index
                self.hive = hive
                self.parameters = parameters

                self.colors_dict = {
                0 : ((255, 255, 255), (200, 200, 200)),
                1 : ((225, 83, 94), (198, 47, 71)),
                2 : ((  0, 190,   0), (  0, 130,   0)),
                3 : ((150, 149, 175), (130, 83, 170)),
                4 : ((248, 204, 0), (249, 170, 14)),
                5 : ((89, 206, 228), (55, 141, 194)),
                6 : ((225, 64, 201), (172, 56, 191)),
                7 : ((133, 90, 83), (117, 54, 42)),
                8 : (( 10, 120, 14), ( 10, 82, 14))
                }
                self.balance_against = {
                0 : 0,
                1 : 0,
                2 : 0,
                3 : 0,
                4 : 0,
                5 : 0,
                6 : 0,
                7 : 0,
                8 : 0
                }
                self.color = 0

                self.forbidden_move = random.choice(("e", "w", "sw", "se", "nw", "ne"))
                self.possible_moves = ["e", "w", "sw", "se", "nw", "ne"]

            # Draw the animal on the screen.
            def draw(self):
                pygame.draw.line(screen, self.colors_dict[self.hive][1], (grid_hex[self.coord_y][self.coord_x][0]+2, grid_hex[self.coord_y][self.coord_x][1]), (grid_hex[self.coord_y][self.coord_x][0]+2+4, grid_hex[self.coord_y][self.coord_x][1]), 1)
                pygame.draw.line(screen, self.colors_dict[self.hive][1], (grid_hex[self.coord_y][self.coord_x][0]+1-1, grid_hex[self.coord_y][self.coord_x][1]+2), (grid_hex[self.coord_y][self.coord_x][0]+1-1, grid_hex[self.coord_y][self.coord_x][1]+6), 1)
                pygame.draw.line(screen, self.colors_dict[self.hive][1], (grid_hex[self.coord_y][self.coord_x][0]+1+7, grid_hex[self.coord_y][self.coord_x][1]+2), (grid_hex[self.coord_y][self.coord_x][0]+1+7, grid_hex[self.coord_y][self.coord_x][1]+6), 1)
                pygame.draw.line(screen, self.colors_dict[self.hive][1], (grid_hex[self.coord_y][self.coord_x][0]+2, grid_hex[self.coord_y][self.coord_x][1]+8), (grid_hex[self.coord_y][self.coord_x][0]+2+4, grid_hex[self.coord_y][self.coord_x][1]+8), 1)
                pygame.draw.rect(screen, self.colors_dict[self.hive][0], [grid_hex[self.coord_y][self.coord_x][0]+1, grid_hex[self.coord_y][self.coord_x][1]+1, 7, 7])

                pygame.draw.circle(screen, self.colors_dict[self.hive][1], (grid_hex[self.coord_y][self.coord_x][0]+1, grid_hex[self.coord_y][self.coord_x][1]+1), 0)
                pygame.draw.circle(screen, self.colors_dict[self.hive][1], (grid_hex[self.coord_y][self.coord_x][0]+7, grid_hex[self.coord_y][self.coord_x][1]+1), 0)
                pygame.draw.circle(screen, self.colors_dict[self.hive][1], (grid_hex[self.coord_y][self.coord_x][0]+1, grid_hex[self.coord_y][self.coord_x][1]+7), 0)
                pygame.draw.circle(screen, self.colors_dict[self.hive][1], (grid_hex[self.coord_y][self.coord_x][0]+7, grid_hex[self.coord_y][self.coord_x][1]+7), 0)
                pygame.draw.circle(screen, GRAY, (grid_hex[self.coord_y][self.coord_x][0]+6, grid_hex[self.coord_y][self.coord_x][1]+2), 0)
                pygame.draw.circle(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2+5, grid_hex[self.coord_y][self.coord_x][1]+8-8), 0)

                # Draw its border.
                pygame.draw.line(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2, grid_hex[self.coord_y][self.coord_x][1]-1), (grid_hex[self.coord_y][self.coord_x][0]+2+4, grid_hex[self.coord_y][self.coord_x][1]-1), 1)
                pygame.draw.line(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+1-2, grid_hex[self.coord_y][self.coord_x][1]+2), (grid_hex[self.coord_y][self.coord_x][0]+1-2, grid_hex[self.coord_y][self.coord_x][1]+6), 1)
                pygame.draw.line(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+1+10-2, grid_hex[self.coord_y][self.coord_x][1]+2), (grid_hex[self.coord_y][self.coord_x][0]+1+10-2, grid_hex[self.coord_y][self.coord_x][1]+6), 1)
                pygame.draw.line(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2, grid_hex[self.coord_y][self.coord_x][1]+8+1), (grid_hex[self.coord_y][self.coord_x][0]+2+4, grid_hex[self.coord_y][self.coord_x][1]+8+1), 1)
                pygame.draw.circle(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2-2, grid_hex[self.coord_y][self.coord_x][1]+8-7), 0)
                pygame.draw.circle(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2-1, grid_hex[self.coord_y][self.coord_x][1]+8-8), 0)
                pygame.draw.circle(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2+5, grid_hex[self.coord_y][self.coord_x][1]+8-8), 0)
                pygame.draw.circle(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2+6, grid_hex[self.coord_y][self.coord_x][1]+8-7), 0)
                pygame.draw.circle(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2+5, grid_hex[self.coord_y][self.coord_x][1]+8+0), 0)
                pygame.draw.circle(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2+6, grid_hex[self.coord_y][self.coord_x][1]+8-1), 0)
                pygame.draw.circle(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2-1, grid_hex[self.coord_y][self.coord_x][1]+8+0), 0)
                pygame.draw.circle(screen, DARKERGRAY, (grid_hex[self.coord_y][self.coord_x][0]+2-2, grid_hex[self.coord_y][self.coord_x][1]+8-1), 0)

            # Move the animal in the positions list and change its coordinates.
            def move(self):
                if int(counter_prev) != int(counter):
                    if int(counter) % 4 == 0:  # self.speed
                        carnivores_pos_hex[self.coord_y][self.coord_x] = \
                            carnivores_pos_hex[self.coord_y][self.coord_x][1:]

                        pre_previous_coord_x = self.previous_coord_x
                        pre_previous_coord_y = self.previous_coord_y

                        self.previous_coord_x = self.coord_x
                        self.previous_coord_y = self.coord_y

                        if not [self.coord_x, self.coord_y] in boundary_tiles:
                            self.possible_moves.remove(self.forbidden_move)  # hash it if you want random movement
                            move = random.choice(self.possible_moves)
                            print(self.index, move, self.hive, " |||||", self.possible_moves, "|||||", self.forbidden_move)
                            if self.coord_y > int(diameter/2):
                                if move == "e":
                                    self.coord_x += 1
                                    self.forbidden_move = "w"
                                elif move == "w":
                                    self.coord_x -= 1
                                    self.forbidden_move = "e"
                                elif move == "sw":
                                    self.coord_x -= 1
                                    self.coord_y += 1
                                    self.forbidden_move = "ne"
                                elif move == "se":
                                    self.coord_y += 1
                                    self.forbidden_move = "nw"
                                elif move == "ne":
                                    self.coord_x += 1
                                    self.coord_y -= 1
                                    self.forbidden_move = "sw"
                                elif move == "nw":
                                    self.coord_y -= 1
                                    self.forbidden_move = "se"
                            elif self.coord_y < int(diameter/2):
                                if move == "e":
                                    self.coord_x += 1
                                    self.forbidden_move = "w"
                                elif move == "w":
                                    self.coord_x -= 1
                                    self.forbidden_move = "e"
                                elif move == "sw":
                                    self.coord_y += 1
                                    self.forbidden_move = "ne"
                                elif move == "se":
                                    self.coord_y += 1
                                    self.coord_x += 1
                                    self.forbidden_move = "nw"
                                elif move == "ne":
                                    self.coord_y -= 1
                                    self.forbidden_move = "sw"
                                elif move == "nw":
                                    self.coord_y -= 1
                                    self.coord_x -= 1
                                    self.forbidden_move = "se"
                            else:  # self.coord_y == diameter-1
                                if move == "e":
                                    self.coord_x += 1
                                    self.forbidden_move = "w"
                                elif move == "w":
                                    self.coord_x -= 1
                                    self.forbidden_move = "e"
                                elif move == "sw":
                                    self.coord_x -= 1
                                    self.coord_y += 1
                                    self.forbidden_move = "ne"
                                elif move == "se":
                                    self.coord_y += 1
                                    self.forbidden_move = "nw"
                                elif move == "ne":
                                    self.coord_y -= 1
                                    self.forbidden_move = "sw"
                                elif move == "nw":
                                    self.coord_x -= 1
                                    self.coord_y -= 1
                                    self.forbidden_move = "se"

                            if self.coord_x == pre_previous_coord_x and self.coord_y == pre_previous_coord_y:
                                print("####### WLASNIE ZAPOBIEGNIETO COFNIECIU SIE####################################################")
                                self.coord_x = self.previous_coord_x
                                self.coord_y = self.previous_coord_y
                        else:
                            self.coord_x = pre_previous_coord_x
                            self.coord_y = pre_previous_coord_y

                            # how to forbid from moving to previous tile?


                        self.possible_moves = ["e", "w", "sw", "se", "nw", "ne"]
                        try:
                            carnivores_pos_hex[self.coord_y][self.coord_x].append(1)
                        except:
                            print(self.coord_y, self.coord_x)
                            time.sleep(1000000)


        class ControllableAgent(Carnivore):
            move_to_do = ""

            def __init__(self):
                super(ControllableAgent, self).__init__(38, 24, -1, 0, [random.randint(-3,3),
                                                    random.randint(-3,3),
                                                    random.randint(-3,3),
                                                    random.randint(-3,3)])

            def move(self):
                #carnivores_pos_hex[self.coord_y][self.coord_x] = \
                    #carnivores_pos_hex[self.coord_y][self.coord_x][1:]

                if self.coord_y > int(diameter/2):
                    if self.move_to_do == "e":
                        self.coord_x += 1
                        self.forbidden_move = "w"
                    elif self.move_to_do == "w":
                        self.coord_x -= 1
                        self.forbidden_move = "e"
                    elif self.move_to_do == "sw":
                        self.coord_x -= 1
                        self.coord_y += 1
                        self.forbidden_move = "ne"
                    elif self.move_to_do == "se":
                        self.coord_y += 1
                        self.forbidden_move = "nw"
                    elif self.move_to_do == "ne":
                        self.coord_x += 1
                        self.coord_y -= 1
                        self.forbidden_move = "sw"
                    elif self.move_to_do == "nw":
                        self.coord_y -= 1
                        self.forbidden_move = "se"
                elif self.coord_y < int(diameter/2):
                    if self.move_to_do == "e":
                        self.coord_x += 1
                        self.forbidden_move = "w"
                    elif self.move_to_do == "w":
                        self.coord_x -= 1
                        self.forbidden_move = "e"
                    elif self.move_to_do == "sw":
                        self.coord_y += 1
                        self.forbidden_move = "ne"
                    elif self.move_to_do == "se":
                        self.coord_y += 1
                        self.coord_x += 1
                        self.forbidden_move = "nw"
                    elif self.move_to_do == "ne":
                        self.coord_y -= 1
                        self.forbidden_move = "sw"
                    elif self.move_to_do == "nw":
                        self.coord_y -= 1
                        self.coord_x -= 1
                        self.forbidden_move = "se"
                else:  # self.coord_y == diameter-1
                    if self.move_to_do == "e":
                        self.coord_x += 1
                        self.forbidden_move = "w"
                    elif self.move_to_do == "w":
                        self.coord_x -= 1
                        self.forbidden_move = "e"
                    elif self.move_to_do == "sw":
                        self.coord_x -= 1
                        self.coord_y += 1
                        self.forbidden_move = "ne"
                    elif self.move_to_do == "se":
                        self.coord_y += 1
                        self.forbidden_move = "nw"
                    elif self.move_to_do == "ne":
                        self.coord_y -= 1
                        self.forbidden_move = "sw"
                    elif self.move_to_do == "nw":
                        self.coord_x -= 1
                        self.coord_y -= 1
                        self.forbidden_move = "se"

                if self.coord_y > len(grid_hex)-1:
                    self.coord_y = 0
                if self.coord_x > len(grid_hex[self.coord_y])-1:
                    self.coord_x = 0
                if self.coord_y < 0:
                    self.coord_y = diameter-1
                if self.coord_x < 0:
                    self.coord_x = len(grid_hex[self.coord_y])-1

                self.move_to_do = ""
                #carnivores_pos_hex[self.coord_y][self.coord_x].append(1)



        # Spawn a new carnivore.
        def spawn_carnivore(amount, hive):
            amount_left_to_spawn = amount
            while amount_left_to_spawn != 0:
                pos_x = random.randint(1, int(grid_size/2))
                pos_y = random.randint(1, grid_size-2)
                if len(carnivores_pos_hex[pos_y][pos_x]) < 1:
                    carnivores.append(Carnivore(pos_x, pos_y, len(carnivores), hive, [
                                                      random.randint(-3,3),
                                                      random.randint(-3,3),
                                                      random.randint(-3,3),
                                                      random.randint(-3,3)]))
                    carnivores_pos_hex[pos_y][pos_x].append(1)
                    amount_left_to_spawn -= 1

        # Spawn a new carnivore but with different parameters setting.
        def spawn_barnivore(amount, hive):
            amount_left_to_spawn = amount
            while amount_left_to_spawn != 0:
                pos_x = random.randint(1, int(grid_size/2))
                pos_y = random.randint(1, grid_size-2)
                if len(carnivores_pos_hex[pos_y][pos_x]) < 1:
                    carnivores.append(Carnivore(pos_x, pos_y, len(carnivores), hive, [
                                                      random.randint(0,1),
                                                      random.randint(0,1),
                                                      random.randint(0,1),
                                                      random.randint(0,1)]))
                    carnivores_pos_hex[pos_y][pos_x].append(1)
                    amount_left_to_spawn -= 1

        # Spawn agents.
        for i in range(0,9):
            spawn_carnivore(nb_agents_in_hives[i], i)

        # Create list of hives present in current simulation.
        hives_present_in_current_sim = []
        for i in range(len(nb_agents_in_hives)):
            if nb_agents_in_hives[i] != 0:
                hives_present_in_current_sim.append(i)
        print("hives_present_in_current_sim:", hives_present_in_current_sim)

        # Create controllable agent
        if debug_controllable_agent_present == 1:
            controllable_agent = ControllableAgent()


        want_rewards_visualisation = 0
        n = self.get_n()  # amount of agent types in the current simulation
        # ################################# #
        # ##-------- PYGAME LOOP --------## #
        # ################################# #

        while not pygame_done:
            # Set cycles per second value.
            clock.tick(60)

            # Catching events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame_done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if pause == 1:
                            pause = 0
                        else:
                            pause = 1
                    if event.key == pygame.K_LEFT:
                        want_rewards_visualisation = 1
                    if event.key == pygame.K_BACKSPACE:
                        if self.exit_after_this_sim == 0:
                            self.exit_after_this_sim = 1
                        else:
                            self.exit_after_this_sim = 0
                    if debug_controllable_agent_present == 1:
                        if event.key == pygame.K_y:
                            controllable_agent.move_to_do = "nw"
                        elif event.key == pygame.K_u:
                            controllable_agent.move_to_do = "ne"
                        elif event.key == pygame.K_j:
                            controllable_agent.move_to_do = "e"
                        elif event.key == pygame.K_n:
                            controllable_agent.move_to_do = "se"
                        elif event.key == pygame.K_b:
                            controllable_agent.move_to_do = "sw"
                        elif event.key == pygame.K_g:
                            controllable_agent.move_to_do = "w"
            # Increase /counter/ with a step of a size of /tempo/ every frame,
            # if simulation isn't paused. If /counter/ is bigger than 120,
            # reset it, and increase /big_counter/ by 1.
            counter_prev = counter
            big_counter_prev = big_counter
            if not pause:
                counter += tempo
            if int(counter) >= 120:
                big_counter += 1
                counter = 0

            # Increase /counter_for_fps/ by 1.
            # If /counter_for_fps/ is bigger than 120,
            # reset it.
            counter_for_fps += 1
            if counter_for_fps == 120:
                counter_for_fps = 0
            # Play all the games that are in the fights_queue
            if not pause:
                if fight_flag == 1:
                    #print(fights_queue)
                    for i in range(len(fights_queue)):
                        episode_reward, episode_step, observation, episode, did_abort = fight(self, env, action_repetition, callbacks, nb_max_start_steps,
                        start_step_policy, nb_max_episode_steps, episode_reward, episode_step, observation, episode, did_abort,
                        fights_queue[i][0], fights_queue[i][1], fights_queue[i][2], fights_queue[i][3], fights_queue[i][4], fights_queue[i][5])
                        print("baba", episode_reward)
                        print("account before:", carnivores[fights_queue[i][2]].account, "::", fights_queue[i][2])
                        carnivores[fights_queue[i][2]].account += episode_reward[0]
                        print("account after:", carnivores[fights_queue[i][2]].account, "::", fights_queue[i][2])
                        print("account1 before:", carnivores[fights_queue[i][3]].account, "::", fights_queue[i][3])
                        carnivores[fights_queue[i][3]].account += episode_reward[1]
                        print("account1 after:", carnivores[fights_queue[i][3]].account, "::", fights_queue[i][3])
                        print("######################################################################")
                        print("######################################################################")

                        # Adding episode rewards to rewards_history.
                        if fights_queue[i][0] == fights_queue[i][1]:
                            self.rewards_history[fights_queue[i][0]][fights_queue[i][1]].append((episode_reward[0] + episode_reward[1])/2)
                        else:
                            self.rewards_history[fights_queue[i][0]][fights_queue[i][1]].append(episode_reward[0])
                            carnivores[fights_queue[i][2]].balance_against[fights_queue[i][1]] += episode_reward[0]
                            hives_balance_against[fights_queue[i][0]][fights_queue[i][1]] += episode_reward[0]
                            hives_balance_against[fights_queue[i][1]][fights_queue[i][0]] += episode_reward[1]
                            self.rewards_history[fights_queue[i][1]][fights_queue[i][0]].append(episode_reward[1])
                            carnivores[fights_queue[i][3]].balance_against[fights_queue[i][0]] += episode_reward[1]

                    fights_queue = []
                    fight_flag = 0

            # Draw interface.
            if counter_for_fps % cycles_per_sec_dividers_list[chosen_cycles_per_second] == 0:
                screen.fill(LIGHTGRAY)
                draw_window()

                # Animation to prevent Windows from hanging the window when paused.
                # Also useful in approximating lag.
                text_to_blit = font7.render(animation[0], True, (50, 50, 50))
                screen.blit(text_to_blit, (847, 651))
                animation = animation + animation[0]
                animation = animation[1:]

            # Create reward charts on demand (left arrow button).
            if want_rewards_visualisation == 1:
                # Smaller number first.
                print("1: DQN_one\n2: DQN_two\n3: AlwaysCoop\n4: AlwaysDefect\n5: GRIM\n6: Imperfect TFT\n7: Random\n8: Suspicious TFT\n9: Tit For Tat")
                x = input("\nWhich agent do you want to see fight?: ")
                y = input("Who will be the opponent?: ")
                while x == '' or y == '' or int(x) > n or int(y) > n:
                    print("### Invalid input. ###\n")
                    print("1: DQN_one\n2: DQN_two\n3: AlwaysCoop\n4: AlwaysDefect\n5: GRIM\n6: Imperfect TFT\n7: Random\n8: Suspicious TFT\n9: Tit For Tat")
                    x = input("\nWhich agent do you want to see fight?: ")
                    y = input("Who will be the opponent?: ")
                x = int(x)-1
                y = int(y)-1
                if x > -1 and y > -1 and x < n and y < n:
                    pp.plot(self.rewards_history[x][y])
                    pp.ylim([-120, 480])
                    pp.show()
                want_rewards_visualisation = 0


            """
            # Checking if all players played at least {amount_of_games_to_log} games.
            if counter_for_fps % 180 == 0:
                temp_counter = 0
                for i in hives_present_in_current_sim:
                    for j in hives_present_in_current_sim:
                        if len(self.rewards_history[i][j]) < amount_of_games_to_log and len(self.rewards_history[i][j]) > 0:
                            temp_counter += 1
                if temp_counter == 0:
                    pygame_done = True
            """


            # "eat", then move.
            if not pause:
                for i in carnivores:
                    i.move()
            if debug_controllable_agent_present == 1:
                controllable_agent.move()

            for i in carnivores:
                if counter_for_fps % cycles_per_sec_dividers_list[chosen_cycles_per_second] == 0:
                    i.draw()
            if debug_controllable_agent_present == 1:
                if counter_for_fps % cycles_per_sec_dividers_list[chosen_cycles_per_second] == 0:
                    controllable_agent.draw()

            # check collisions and create fights.
            for j in carnivores:
                if len(carnivores_pos_hex[j.coord_y][j.coord_x]) > 1:
                    for i in carnivores:
                        if j.get_coords()[0] == i.get_coords()[0] and j.get_coords()[1] == i.get_coords()[1]:
                            if int(counter_prev) != int(counter) and int(counter) % 4 == 0:
                                if j.index not in agents_fighting_queue and i.index not in agents_fighting_queue:
                                    if j.index != i.index:
                                        agents_fighting_queue.add(j.index)
                                        agents_fighting_queue.add(i.index)
                                        fights_queue.append([j.hive, i.hive, j.index, i.index, j.parameters, i.parameters])
                                        fight_flag = 1
                                        break
            agents_fighting_queue = set()
            for i in fights_queue:  # swap the order of agents on the list so that smaller number is first (technical stuff)
                if i[0] > i[1]:
                    i[0], i[1] = i[1], i[0]

            # Update the screen.
            pygame.display.flip()

        # At the end of entire simulation, call on_train_end callback.
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        # After the simulation is done, append to .csv
        for i in range(len(self.rewards_history)):
            for j in range(len(self.rewards_history[i])):
                # Open our existing CSV file in append mode.
                # Create a file object for this file.
                file_name = 'csv output/{}.csv'.format([i,j])
                print("file-name:", file_name)
                with open(file_name, 'a') as f_object:
                    # Pass this file object to csv.writer()
                    # and get a writer object.
                    writer_object = writer(f_object)
                    writer_object.writerow(self.rewards_history[i][j][:amount_of_games_to_log])
                    #Close the file object
                    f_object.close()


        # Return history.
        return history





    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.
        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = [0, 0]
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(
                        observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                        nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = [0, 0]
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(
                            observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward[0] += r[0]
                    reward[1] += r[1]

                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward[0] += reward[0]
                episode_reward[1] += reward[1]
                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward([0, 0], terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history

    def test_fight(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1, player = None, player_one = None):
        """Callback that is called before training begins.
        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = [0, 0]
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(
                        observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                        nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward_test(observation, player, player_one, 0, 0)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = [0, 0]
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(
                            observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward[0] += r[0]
                    reward[1] += r[1]

                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward[0] += reward[0]
                episode_reward[1] += reward[1]
                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward_test(observation, player, player_one, 0, 0)
            self.backward([0, 0], terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history


    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.
        # Argument
            observation (object): The current observation from the environment.
        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.
        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        # Returns
            List of metrics values
        """
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.
        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.
        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.
        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).
        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.
        # Returns
            A list of the model's layers
        """
        raise NotImplementedError()

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        # Returns
            A list of metric's names (string)
        """
        return []

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        pass

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        pass


class Processor(object):
    """Abstract base class for implementing processors.
    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.
    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.
        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.
        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            observation (object): An observation as obtained by the environment
        # Returns
            Observation obtained by the environment processed
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            reward (float): A reward as obtained by the environment
        # Returns
            Reward obtained by the environment processed
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            info (dict): An info as obtained by the environment
        # Returns
            Info obtained by the environment processed
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.
        # Arguments
            action (int): Action given to the environment
        # Returns
            Processed action given to the environment
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        # Arguments
            batch (list): List of states
        # Returns
            Processed list of states
        """
        return batch

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.
        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []


# Note: the API of the `Env` and `Space` classes are taken from the OpenAI Gym implementation.
# https://github.com/openai/gym/blob/master/gym/core.py


class Env(object):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    To implement your own environment, you need to define the following methods:
    - `step`
    - `reset`
    - `render`
    - `close`
    Refer to the [Gym documentation](https://gym.openai.com/docs/#environments).
    """
    reward_range = (-np.inf, np.inf)
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        raise NotImplementedError()

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        raise NotImplementedError()

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class Space(object):
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.
    Please refer to [Gym Documentation](https://gym.openai.com/docs/#spaces)
    """

    def sample(self, seed=None):
        """Uniformly randomly sample a random element of this space.
        """
        raise NotImplementedError()

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        raise NotImplementedError()
