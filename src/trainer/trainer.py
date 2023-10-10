from abc import ABC, abstractmethod
import jax
from src.agent.episodic_replay_buffer import EpisodicReplayBuffer

# Define abstract Trainer class
class Trainer(ABC):
    def __init__(self, 
            encoder_fn: callable,
            optimizer: callable, 
            replay_buffer: EpisodicReplayBuffer, 
            logger, 
            rng_key: jax.random.PRNGKey, 
            **kwargs
        ):
        super().__init__()

        # Store model
        self.encoder_fn = encoder_fn
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.rng_key = rng_key

        # Store all keyword arguments as attributes
        self.__dict__.update((k, v) for k, v in kwargs.items())

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def init_additional_params(self, *args, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    # def metrics(self, *args, **kwargs):
    #     raise NotImplementedError

    @abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def additional_update_step(self, *args, **kwargs):
        raise NotImplementedError
