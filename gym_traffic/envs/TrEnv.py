import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym_traffic.envs import traffic_simulator

class TrEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        self.sim = traffic_simulator.TrafficSim()

    def step(self, action):
        self.sim.step(action)
        ob = self._get_obs()
        reward = self.sim.getReward()
        done = False
        return ob, reward, done, reward

    def _get_obs(self):
        return self.sim.getGameState()

    def reset(self):
        self.sim.reset()
        return self.sim.getGameState()
