"""Base mixins for training."""

from abc import abstractmethod, ABC


class ModelTrainerBase(ABC):

    def __init__(
        self, 
        epochs: int, 
        *args, **kwargs,
    ):
        self.epochs = epochs
        self._args = args
        self._kwargs = kwargs
        self._trainer = None

    @abstractmethod
    def create_trainer(self):
        pass

    @abstractmethod
    def train(self, model, train_dataloader, val_dataloader):
        pass
