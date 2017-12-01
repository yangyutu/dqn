from gym.envs.registration import register


register(
    id='TigerProblem-v0',
    entry_point='envs.tiger_problem:TigerProblemEnv',
)
