from abc import ABC, abstractmethod

# Define abstract Trainer class
class Trainer(ABC):
    def __init__(self, model, optimizer, replay_buffer, logger, **kwargs):
        super().__init__()

        # Store model
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.logger = logger

        # Store all keyword arguments as attributes
        self.__dict__.update((k, v) for k, v in kwargs.items())

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def metrics(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError
