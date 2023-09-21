import abc
import random
from src.tools.random import set_random_number_generator

class Policy(abc.ABC):
    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

class DiscreteUniformRandomPolicy(Policy):
    '''
        Policy that generates discrete actions 
        following a uniform distribution.
    '''
    def __init__(
            self, 
            num_actions: int, 
            random_number_generator: random.Random = None,
            seed: int = 0,
        ) -> None:
        # Set action space
        self.num_actions = num_actions
        # Set generator
        set_random_number_generator(self, random_number_generator, seed)

    def act(self, state):
        return self.random_number_generator.randint(0, self.num_actions-1)

    def __str__(self):
        return f"discrete_uniform_random_policy({self.num_actions})"
