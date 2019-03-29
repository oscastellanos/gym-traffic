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
        #self.action_space = Discrete(n=2)
        #self.observation_space = Box(low=0, high=20000, shape=(1,), dtype=np.float32)

    def step(self, action):
        self.sim.step(action)
        ob, reward, done = self._get_obs()
        return ob, reward, done, False

    def _get_obs(self):
        return self.sim.getGameState()

    def reset(self):
        self.sim.reset()
        return self.sim.getGameState()
    
