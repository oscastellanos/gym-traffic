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
        #self.viewer = rendering.Viewer()

    def step(self, action):
        # self.sim.step(action)
        # ob, reward, done, signal = self._get_obs()
        ob, reward, done, signal = self.sim.step(action)
        return ob, reward, done, signal

    def _get_obs(self):
        return self.sim.getGameState()

    def reset(self):
        self.sim.reset()
        return self.sim.getGameState()

    def render(self, mode='human'):
        #from gym_traffic.envs import rendering
        #s = self.state 
        #if self.viewer is None:
        #    self.viewer = rendering.Viewer(500,500)
        return self.sim.render()


        # pygame.init()
        
        # #game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
        # game.clock = pygame.time.Clock()
        # game.rng = np.random.RandomState(24)
        # game.init()

    
