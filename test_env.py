import gymnasium as gym
import rl_lap.env

if __name__ == "__main__":
    path_txt_grid = './rl_lap/env/grid/txts/GridMaze-17.txt'
    env = gym.make('Grid-v0', path=path_txt_grid, render_mode="human", render_fps=50)
    print(env.grid.shape)
    observation, info = env.reset()

    for _ in range(10000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()