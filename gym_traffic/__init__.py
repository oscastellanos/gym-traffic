from gym.envs.registration import register
#
register(
     id='traffic-v1',
     entry_point='gym_traffic.envs:TrEnv',
)
