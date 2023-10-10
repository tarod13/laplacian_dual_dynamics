from gymnasium.envs.registration import register

register(
    id='Grid-v0',
    entry_point='src.env.grid:GridEnv',
    max_episode_steps=300,
)

register(
    id='LapGrid-v0',
    entry_point='src.env.grid:LaplacianGridEnv',
    max_episode_steps=300,
)