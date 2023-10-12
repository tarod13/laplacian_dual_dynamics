import gymnasium as gym
import src.env
from src.env.grid.utils import load_eig

if __name__ == "__main__":
    env_name = 'GridRoom-16'
    path_txt_grid = f'./src/env/grid/txts/{env_name}.txt'
    path_eig = f'./src/env/grid/eigval/{env_name}.npz'

    eig = load_eig(path_eig)[0]
    env = gym.make('Grid-v0', path=path_txt_grid, render_mode="human", render_fps=5, eig=eig, obs_mode='pixels')
    print(env.grid.shape)
    observation, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        if 'pixels' in observation.keys():
            print(observation['pixels'].shape)

        if terminated or truncated:
            observation, info = env.reset()
            
    env.close()