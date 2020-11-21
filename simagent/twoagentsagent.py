# -*- coding: utf-8 -*-
import warnings
from copy import deepcopy

import numpy as np
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
from positions import *



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

        """
        size = (1061, 670)
        pygame.display.set_caption("HIVE")
        screen = pygame.display.set_mode(size)
        pygame.draw.rect(screen, (100,100,100), [50, 60, 162, 105])
        screen.fill((0,0,0))
        clock = pygame.time.Clock()
        clock.tick(15)
        pygame.display.flip()"""

    def get_config(self):
        """Configuration of the agent for serialization.
        # Returns
            Dictionnary with agent configuration
        """
        return {}

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
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
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False


        pygame.init()
        size = (1061, 670)
        pygame.display.set_caption("HIVE")
        screen = pygame.display.set_mode(size)
        # Set the main loop to not done and initialize a clock.
        pygame_done = False
        clock = pygame.time.Clock()


##########################################################################
        def fight(self, env, action_repetition, callbacks, nb_max_start_steps, start_step_policy,
        nb_max_episode_steps, episode_reward, episode_step, observation, episode, did_abort, player = None, player_one = None):
            try:
                #while self.step < nb_steps:
                for i_step in range(20):
                    if observation is None:  # start of a new episode
                        callbacks.on_episode_begin(episode)
                        episode_step = np.int16(0)
                        episode_reward = [0, 0] # ZMIANA #

                        # Obtain the initial observation by resetting the environment.
                        self.reset_states()
                        observation = deepcopy(env.reset())
                        if self.processor is not None:
                            observation = self.processor.process_observation(
                                observation)
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
                            observation, reward, done, info = env.step(action)
                            observation = deepcopy(observation)
                            if self.processor is not None:
                                observation, reward, done, info = self.processor.process_step(
                                    observation, reward, done, info)
                            callbacks.on_action_end(action)
                            if done:
                                warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                                    nb_random_start_steps))
                                observation = deepcopy(env.reset())
                                if self.processor is not None:
                                    observation = self.processor.process_observation(
                                        observation)
                                break

                    # At this point, we expect to be fully initialized.
                    assert episode_reward is not None
                    assert episode_step is not None
                    assert observation is not None

                    # Run a single step.
                    callbacks.on_step_begin(episode_step)
                    # This is where all of the work happens. We first perceive and compute the action
                    # (forward step) and then use the reward to improve (backward step).
                    action = self.forward_test(observation, player, player_one)
                    if self.processor is not None:
                        action = self.processor.process_action(action)

                    reward = [0, 0] # ZMIANA #
                    accumulated_info = {}
                    done = False
                    for _ in range(action_repetition):
                        callbacks.on_action_begin(action)
                        observation, r, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, r, done, info = self.processor.process_step(
                                observation, r, done, info)
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
                    if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                        # Force a terminal state.
                        done = True

                    metrics, metrics1 = self.backward(reward, terminal=done)

                    # ZMIANA #

                    #episode_reward.append([reward[0][0], reward[0][1]])
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
                        self.forward_test(observation, player, player_one)
                        self.backward([0, 0], terminal=False)

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
                        episode_reward = None
            except KeyboardInterrupt:
                # We catch keyboard interrupts here so that training can be be safely aborted.
                # This is so common that we've built this right into this function, which ensures that
                # the `on_train_end` method is properly called.
                did_abort = True

            return episode_reward, episode_step, observation, episode, did_abort

##########################################################################
##########################################################################
##########################################################################

        font  = pygame.font.SysFont("liberationmono", 13)
        font2 = pygame.font.SysFont("liberationmono", 12)
        font3 = pygame.font.SysFont("liberationmono", 11)
        font4 = pygame.font.SysFont("humorsans", 70)
        font5 = pygame.font.SysFont("liberationmono", 12)
        font6 = pygame.font.SysFont("liberationmono", 14)
        font7 = pygame.font.SysFont("liberationmono", 18)

        # Define colors.
        colors_list_red = [
        [(255, 173, 153), (255, 153, 128), (255, 133, 102), (255, 112, 77), (255, 92, 51), (255, 71, 26), (255, 51, 0), (230, 46, 0)],            # RED 1
        [(255, 194, 153), (255, 179, 128), (255, 163, 102), (255, 148, 77), (255, 133, 51), (255, 117, 26), (255, 102, 0), (230, 92, 0)],         # RED 2
        [(255, 153, 153), (255, 128, 128), (255, 102, 102), (255, 77, 77), (255, 51, 51), (255, 26, 26), (255, 0, 0), (230, 0, 0)],               # RED 3
        [(255, 153, 187), (255, 128, 170), (255, 102, 153), (255, 77, 136), (255, 51, 119), (255, 26, 102), (255, 0, 85), (230, 0, 76)],          # RED 4
        [(255, 173, 153), (255, 153, 128), (255, 133, 102), (255, 112, 77), (255, 92, 51), (255, 71, 26), (255, 51, 0), (230, 46, 0)],            # RED 1
        [(255, 194, 153), (255, 179, 128), (255, 163, 102), (255, 148, 77), (255, 133, 51), (255, 117, 26), (255, 102, 0), (230, 92, 0)],         # RED 2
        [(255, 153, 153), (255, 128, 128), (255, 102, 102), (255, 77, 77), (255, 51, 51), (255, 26, 26), (255, 0, 0), (230, 0, 0)],               # RED 3
        [(255, 153, 187), (255, 128, 170), (255, 102, 153), (255, 77, 136), (255, 51, 119), (255, 26, 102), (255, 0, 85), (230, 0, 76)]           # RED 4
        ]
        colors_list_green = [
        [(77, 255, 136), (51, 255, 119), (26, 255, 102), (0, 255, 85), (0, 230, 77), (0, 204, 68), (0, 179, 60), (0, 153, 51), (0, 128, 43)],     # GREEN 1
        [(77, 255, 77), (51, 255, 51), (26, 255, 26), (153,50,204), (0, 230, 0), (0, 204, 0), (0, 179, 0), (153,0,204), (0, 128, 0)],              # PURPLE
        [(51, 255, 153), (26, 255, 140), (0, 255, 128), (0, 230, 115), (0, 204, 102), (0, 179, 89), (0, 153, 77), (0, 128, 64)],                  # GREEN 3
        [(51, 255, 204), (26, 255, 198), (0, 255, 191), (0, 230, 172), (0, 204, 153), (0, 179, 134), (0, 153, 115), (0, 128, 96)],                # GREEN 4
        [(77, 255, 136), (51, 255, 119), (26, 255, 102), (0, 255, 85), (0, 230, 77), (0, 204, 68), (0, 179, 60), (0, 153, 51), (0, 128, 43)],     # GREEN 1
        [(77, 255, 77), (51, 255, 51), (26, 255, 26), (0, 255, 0), (0, 230, 0), (0, 204, 0), (0, 179, 0), (0, 153, 0), (0, 128, 0)],              # GREEN 2
        [(51, 255, 153), (26, 255, 140), (0, 255, 128), (0, 230, 115), (0, 204, 102), (0, 179, 89), (0, 153, 77), (0, 128, 64)],                  # GREEN 3
        [(51, 255, 204), (26, 255, 198), (0, 255, 191), (0, 230, 172), (0, 204, 153), (0, 179, 134), (0, 153, 115), (0, 128, 96)]                 # GREEN 4
        ]

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
        PLUS_UP = pygame.image.load("sprites/PLUS_UP.png")
        PLUS_DOWN = pygame.image.load("sprites/PLUS_DOWN.png")
        MINUS_UP = pygame.image.load("sprites/MINUS_UP.png")
        MINUS_DOWN = pygame.image.load("sprites/MINUS_DOWN.png")
        RESET_UP = pygame.image.load("sprites/RESET_UP.png")
        RESET_DOWN = pygame.image.load("sprites/RESET_DOWN.png")
        START_UP = pygame.image.load("sprites/START_UP.png")
        START_DOWN = pygame.image.load("sprites/START_DOWN.png")
        PAUSE_UP = pygame.image.load("sprites/PAUSE_UP.png")
        PAUSE_DOWN = pygame.image.load("sprites/PAUSE_DOWN.png")
        RIGHT_PANEL_BUTTONS1 = pygame.image.load("sprites/RIGHT_PANEL_BUTTONS1.png")
        RIGHT_PANEL_BUTTONS2 = pygame.image.load("sprites/RIGHT_PANEL_BUTTONS2.png")
        RIGHT_PANEL_BUTTONS3 = pygame.image.load("sprites/RIGHT_PANEL_BUTTONS3.png")

        animation = "|/-\\"

        carnivores = []
        btn_tempo_plus_clicked = 0
        btn_tempo_minus_clicked = 0

        tempo = 1                      # between 0.01 and 1    # default = 0.28
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
        grid_size = 43 # max amount of agents = (grid_size-2)^2, if you exceed, they won't have space to spawn, max 43
        fight_flag = 0

        def draw_window():
            signature = font3.render("bsski 2020", True, (200, 200, 200))
            screen.blit(signature, (70, 60))

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

            # Charts backgrounds.
            pygame.draw.rect(screen, WHITE, [29, 124, 162, 300])
            pygame.draw.rect(screen, GRAY, [860, 44, 176, 6])
            pygame.draw.rect(screen, GRAY, [850, 50, 196, 609])
            #if bigger_screen == 1:
            pygame.draw.rect(screen, WHITE, [37, 459, 801, 191])

            # Main interface lines.
            pygame.draw.line(screen, GRAY, (12, 12), (12, 657), 1)
            pygame.draw.line(screen, GRAY, (198, 12), (198, 436), 1)
            pygame.draw.line(screen, GRAY, (656, 12), (656, 436), 1)
            pygame.draw.line(screen, GRAY, (847, 12), (847, 657), 1)
            pygame.draw.line(screen, GRAY, (1048, 12), (1048, 646), 1)

            # Amounts chart.
            pygame.draw.line(screen, GRAY, (28, 423), (189, 423), 1)
            pygame.draw.line(screen, GRAY, (28, 413), (189, 413), 1)
            pygame.draw.line(screen, GRAY, (28, 403), (189, 403), 1)
            pygame.draw.line(screen, GRAY, (28, 393), (189, 393), 1)
            pygame.draw.line(screen, GRAY, (28, 383), (189, 383), 1)
            pygame.draw.line(screen, GRAY, (28, 373), (189, 373), 1)
            pygame.draw.line(screen, GRAY, (28, 363), (189, 363), 1)
            pygame.draw.line(screen, GRAY, (28, 353), (189, 353), 1)
            pygame.draw.line(screen, GRAY, (28, 343), (189, 343), 1)
            pygame.draw.line(screen, GRAY, (28, 333), (189, 333), 1)
            pygame.draw.line(screen, GRAY, (28, 323), (189, 323), 1)
            pygame.draw.line(screen, GRAY, (28, 313), (189, 313), 1)
            pygame.draw.line(screen, GRAY, (28, 303), (189, 303), 1)
            pygame.draw.line(screen, GRAY, (28, 293), (189, 293), 1)
            pygame.draw.line(screen, GRAY, (28, 283), (189, 283), 1)
            pygame.draw.line(screen, GRAY, (28, 273), (189, 273), 1)
            pygame.draw.line(screen, GRAY, (28, 263), (189, 263), 1)
            pygame.draw.line(screen, GRAY, (28, 253), (189, 253), 1)
            pygame.draw.line(screen, DARKRED, (28, 243), (189, 243), 1)
            pygame.draw.line(screen, GRAY, (28, 233), (189, 233), 1)
            pygame.draw.line(screen, GRAY, (28, 223), (189, 223), 1)
            pygame.draw.line(screen, GRAY, (28, 213), (189, 213), 1)
            pygame.draw.line(screen, GRAY, (28, 203), (189, 203), 1)
            pygame.draw.line(screen, GRAY, (28, 193), (189, 193), 1)
            pygame.draw.line(screen, GRAY, (28, 183), (189, 183), 1)
            pygame.draw.line(screen, GRAY, (28, 173), (189, 173), 1)
            pygame.draw.line(screen, GRAY, (28, 163), (189, 163), 1)
            pygame.draw.line(screen, GRAY, (28, 153), (189, 153), 1)
            pygame.draw.line(screen, GRAY, (28, 143), (189, 143), 1)
            pygame.draw.line(screen, GRAY, (28, 133), (189, 133), 1)
            pygame.draw.line(screen, GRAY, (28, 123), (189, 123), 1)
            pygame.draw.line(screen, DARKGRAY, (28, 424), (28, 122), 1)
            pygame.draw.line(screen, DARKGRAY, (28, 424), (189, 424), 1)
            pygame.draw.line(screen, DARKGRAY, (28, 122), (189, 122), 1)
            pygame.draw.line(screen, DARKGRAY, (190, 424), (190, 122), 1)
            text_to_blit = font5.render("0", True, (50, 50, 50))
            screen.blit(text_to_blit, (18, 417))
            text_to_blit = font5.render(".1", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 397))
            text_to_blit = font5.render(".2", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 377))
            text_to_blit = font5.render(".3", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 357))
            text_to_blit = font5.render(".4", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 337))
            text_to_blit = font5.render(".5", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 317))
            text_to_blit = font5.render(".6", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 297))
            text_to_blit = font5.render(".7", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 277))
            text_to_blit = font5.render(".8", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 257))
            text_to_blit = font5.render(".9", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 237))
            text_to_blit = font5.render("1", True, (50, 50, 50))
            screen.blit(text_to_blit, (17, 217))
            text_to_blit = font5.render(".1", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 197))
            text_to_blit = font5.render(".2", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 177))
            text_to_blit = font5.render(".3", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 157))
            text_to_blit = font5.render(".4", True, (50, 50, 50))
            screen.blit(text_to_blit, (12, 137))
            text_to_blit = font5.render("k", True, (50, 50, 50))
            screen.blit(text_to_blit, (18, 117))

            # All amount history chart.
            #if bigger_screen == 1:
            pygame.draw.line(screen, GRAY, (37, 649), (837, 649), 1)
            pygame.draw.line(screen, GRAY, (37, 639), (837, 639), 1)
            pygame.draw.line(screen, GRAY, (37, 629), (837, 629), 1)
            pygame.draw.line(screen, GRAY, (37, 619), (837, 619), 1)
            pygame.draw.line(screen, GRAY, (37, 609), (837, 609), 1)
            pygame.draw.line(screen, GRAY, (37, 599), (837, 599), 1)
            pygame.draw.line(screen, GRAY, (37, 589), (837, 589), 1)
            pygame.draw.line(screen, GRAY, (37, 579), (837, 579), 1)
            pygame.draw.line(screen, GRAY, (37, 569), (837, 569), 1)
            pygame.draw.line(screen, GRAY, (37, 559), (837, 559), 1)
            pygame.draw.line(screen, GRAY, (37, 549), (837, 549), 1)
            pygame.draw.line(screen, GRAY, (37, 539), (837, 539), 1)
            pygame.draw.line(screen, GRAY, (37, 529), (837, 529), 1)
            pygame.draw.line(screen, GRAY, (37, 519), (837, 519), 1)
            pygame.draw.line(screen, GRAY, (37, 509), (837, 509), 1)
            pygame.draw.line(screen, GRAY, (37, 499), (837, 499), 1)
            pygame.draw.line(screen, GRAY, (37, 489), (837, 489), 1)
            pygame.draw.line(screen, GRAY, (37, 479), (837, 479), 1)
            pygame.draw.line(screen, DARKRED, (37, 469), (837, 469), 1)
            pygame.draw.line(screen, GRAY, (37, 459), (837, 459), 1)
            pygame.draw.line(screen, DARKGRAY, (36, 649+1), (36, 459-1), 1)
            pygame.draw.line(screen, DARKGRAY, (36, 649+1), (837, 649+1), 1)
            pygame.draw.line(screen, DARKGRAY, (36, 459-1), (837, 459-1), 1)
            pygame.draw.line(screen, DARKGRAY, (838, 649+1), (838, 459-1), 1)
            text_to_blit = font2.render("TOTAL AMOUNT LOG", True, (50, 50, 50))
            screen.blit(text_to_blit, (370, 653))

            # Herbs icon.
            pygame.draw.circle(screen, FORESTGREEN, [711, 22], 3, 0)
            # Herbivores icon.
            pygame.draw.rect(screen, colors_list_green[7][7], [684 , 33, 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [684-1, 33-1, 11, 11], 1)
            # Carnivores icon.
            pygame.draw.rect(screen, colors_list_red[7][7], [684, 48, 9, 9])
            pygame.draw.rect(screen, DARKERGRAY, [684-1, 48-1, 11, 11], 1)


        # Class creating animals.
        class animal:
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
        class Carnivore(animal):
            def __init__(self, coord_x, coord_y, index, hive):
                self.coord_x = coord_x
                self.coord_y = coord_y
                self.index = index
                self.hive = hive
                self.colors_dict = {
                0: (255,255,255),
                1:(255,0,0),
                2:(0,255,0),
                3:(0,0,255),
                4:(255,255,0),
                5:(0,255,255),
                6:(255,0,255),
                7:(128,128,0),
                8:(0,128,0)
                }
                self.color = 0
                self.forbidden_move = random.choice(("e", "w", "s", "n"))
                self.possible_moves = ["e", "w", "s", "n"]

            def draw(self):
                pygame.draw.rect(screen, self.colors_dict[self.hive],
                                     [grid[self.coord_y][self.coord_x][0]+1,
                                      grid[self.coord_y][self.coord_x][1]+1, 7, 7])
                # Draw its border.
                pygame.draw.rect(screen, DARKERGRAY,
                                 [grid[self.coord_y][self.coord_x][0],
                                  grid[self.coord_y][self.coord_x][1], 9, 9], 1)

            def move(self):
                if int(counter_prev) != int(counter):
                    if int(counter) % 4 == 0:  # self.speed
                        carnivores_pos[self.coord_y][self.coord_x] = \
                            carnivores_pos[self.coord_y][self.coord_x][1:]

                        if not (self.coord_x == 0 or
                                self.coord_x == grid_size-1 or
                                self.coord_y == 0 or
                                self.coord_y == grid_size-1):
                                # If all conditions satisfied:
                            self.possible_moves.remove(self.forbidden_move) # hash it if you want random movement
                            move = random.choice(self.possible_moves)
                            if move == "e":
                                self.coord_x += 1
                                self.forbidden_move = "w"
                            elif move == "w":
                                self.coord_x -= 1
                                self.forbidden_move = "e"
                            elif move == "s":
                                self.coord_y += 1
                                self.forbidden_move = "n"
                            elif move == "n":
                                self.coord_y -= 1
                                self.forbidden_move = "s"
                        else:
                            if self.coord_x == 0:
                                self.coord_x += 1
                                self.forbidden_move = "w"
                            elif self.coord_x == grid_size-1:
                                self.coord_x -= 1
                                self.forbidden_move = "e"
                            elif self.coord_y == 0:
                                self.coord_y += 1
                                self.forbidden_move = "n"
                            elif self.coord_y == grid_size-1:
                                self.coord_y -= 1
                                self.forbidden_move = "s"
                        self.possible_moves = ["e", "w", "s", "n"]
                        carnivores_pos[self.coord_y][self.coord_x].append(1)


        # Spawn a new carnivore.
        def spawn_carnivore(amount, hive):
            amount_left_to_spawn = amount
            while amount_left_to_spawn != 0:
                pos_y = random.randint(1, grid_size-2)
                pos_x = random.randint(1, grid_size-2)
                if len(carnivores_pos[pos_y][pos_x]) < 1:
                    carnivores.append(Carnivore(pos_x, pos_y, len(carnivores), hive))
                    carnivores_pos[pos_y][pos_x].append(1)
                    amount_left_to_spawn -= 1

        for i in range(0,8):
            spawn_carnivore(4,i)

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
                        if not pause:

                            fights_queue.append([0, 3])
                            print(fights_queue)
                            fight_flag = 1

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
                    print(fights_queue)
                    for i in range(len(fights_queue)):
                        episode_reward, episode_step, observation, episode, did_abort = fight(self, env, action_repetition, callbacks, nb_max_start_steps, start_step_policy,
                        nb_max_episode_steps, episode_reward, episode_step, observation, episode, did_abort, fights_queue[i][0], fights_queue[i][1])
                    fights_queue = []
                    fight_flag = 0

            # Draw interface.
            if counter_for_fps % cycles_per_sec_dividers_list[chosen_cycles_per_second] == 0:
                screen.fill(LIGHTGRAY)
                draw_window()
                # Animation to prevent Windows from hanging the window when paused.
                # Also useful in approximating lag.
                text_to_blit = font7.render(animation[0], True, (50, 50, 50))
                screen.blit(text_to_blit, (1048, 648))
                animation = animation + animation[0]
                animation = animation[1:]

            # "eat", then move.
            if not pause:
                for i in carnivores:
                    i.move()
            for i in carnivores:
                if counter_for_fps % cycles_per_sec_dividers_list[chosen_cycles_per_second] == 0:
                    i.draw()


            # check collisions
            for j in carnivores:
                if len(carnivores_pos[j.coord_y][j.coord_x]) > 1:
                    for i in carnivores:
                        if j.get_coords()[0] == i.get_coords()[0] and j.get_coords()[1] == i.get_coords()[1]:
                            if int(counter_prev) != int(counter) and int(counter) % 4 == 0:
                                if j.index not in agents_fighting_queue and i.index not in agents_fighting_queue:
                                    if j.index != i.index:
                                        agents_fighting_queue.add(j.index)
                                        agents_fighting_queue.add(i.index)
                                        fights_queue.append([j.hive, i.hive])
                                        fight_flag = 1
                                        break
            agents_fighting_queue = set()
            for i in fights_queue:
                if i[0] > i[1]:
                    i[0], i[1] = i[1], i[0]


            # Update the screen.
            pygame.display.flip()

        # At the end of entire simulation, call on_train_end callback.
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

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

                action = self.forward_test(observation, player, player_one)
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
            self.forward_test(observation, player, player_one)
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
