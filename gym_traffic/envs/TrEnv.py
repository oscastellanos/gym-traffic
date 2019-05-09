import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym_traffic.envs import traffic_simulator
import pygame

class TrEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        self.sim = traffic_simulator.TrafficSim()
        self.action_space = spaces.Discrete(n=2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,7), dtype=np.uint8)
        self.viewer = None

    def step(self, action):
        # self.sim.step(action)
        # ob, reward, done, signal = self._get_obs()
        ob, reward, done, signal = self.sim.step(action)
        return ob, reward, done, signal

    def _get_obs(self):
        return self.sim.getGameState()

    def reset(self):
        self.sim.reset()
        return self.sim.getGameState()[0]

    def _reset(self):
        self.sim.reset()
        return self.sim.getGameState()

    def _get_image(self):
        img = self.sim.getScreenRGB()
        return img

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        img_rotated = np.fliplr(np.rot90(np.rot90(np.rot90(img)))) 

        if mode == 'rgb_array':
            return img_rotated
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img_rotated)
        #return self.sim.render()


        # pygame.init()
        
        # #game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
        # game.clock = pygame.time.Clock()
        # game.rng = np.random.RandomState(24)
        # game.init()

    
