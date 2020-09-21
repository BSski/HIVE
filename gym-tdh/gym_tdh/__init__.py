from gym.envs.registration import register

register(
    id='Tdh-v0',
    entry_point='gym_tdh.envs:TdhEnv',
)
