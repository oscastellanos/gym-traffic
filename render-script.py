import gym
import gym_traffic
import numpy as np


env = gym.make('traffic-v1')
x = 0
env.reset()
while (x < 100):
    env.step(env.action_space.sample())
    #print(env.render(mode='rgb_array'))
    #env.render(mode='human')
    x = x + 1
