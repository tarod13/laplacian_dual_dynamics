import abc
import collections
import gymnasium as gym

from src.policy import Policy

Step = collections.namedtuple('Step', 'agent_state, action, episode_done')


class Agent(abc.ABC):
    def __init__(self, policy: Policy):
        self.policy = policy
        self.state = None

    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

class BehaviorAgent(Agent):
    def __init__(self, policy: Policy):
        super().__init__(policy)

    def act(self, state):
        action = self.policy.act(state)
        return action
    
    def infer_state(self, observation, episode_done):
        self.state = observation
    
    def collect_experience(
            self, 
            env: gym.Env, 
            num_steps: int
        ) -> list:
        steps = []
        observation = env.reset()[0]
        episode_done = False
        for _ in range(num_steps):
            # Get action and store current state
            self.infer_state(observation, episode_done)
            agent_state = self.state
            action = self.act(agent_state)
            step = Step(agent_state, action, episode_done)
            steps.append(step)
            
            if episode_done:
                # Reset environment if episode is done
                observation = env.reset()[0]
                episode_done = False
            else:
                # Take action and get next observation
                observation, reward, terminated, truncated, info = env.step(action)
                episode_done = terminated or truncated

        return steps

    def __str__(self):
        return f"behavior_agent({str(self.policy)})"