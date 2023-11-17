import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gymnasium as gym


if __name__ == "__main__":
    env_name = 'ALE/MontezumaRevenge-v5'
    save_path = f'./results/visuals/atari/{env_name[4:-3]}/trajectories.npy'
    obs_list = []
    
    # Create environment
    env = gym.make(
        env_name,
        render_mode="human",
    )

    # Generate observations
    observation, info = env.reset()
    obs_list = [observation]

    for i in range(100):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        obs_list.append(observation)

        if terminated or truncated:
            observation, info = env.reset()
            obs_list.append(observation)

    env.close()

    # Create path if it does not exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Save observations

    with open(save_path, 'wb') as file:
        observations = np.stack(obs_list, axis=0)
        np.save(file, observations)