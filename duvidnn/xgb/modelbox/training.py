"""Mixins for training."""

import numpy as np
from numpy.typing import ArrayLike

try:
    import xgboost as xgb
except ImportError:
    from carabiner import print_err
    print_err(
        """
        [ERROR] XGBoost not installed! Try:
            $ pip install duvidnn[xgb]
        """
    )
    sys.exit(1)

from ...base.training import ModelTrainerBase


class XGBTrainer:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
    
    @staticmethod
    def fit(model, train_dataloader, val_dataloader):
        booster = xgboost.train(
            {"tree_method": "hist"}, 
            train_dataloader,
            xgb_model=model,
        )


class ModelTrainer(ModelTrainerBase):

    _trainer: Trainer = None

    def create_trainer(self) -> None:
        kwargs = {} | self._kwargs
        self._trainer = XGBTrainer(
            **kwargs,
        )
        return None

    def train(
        self, 
        model: LightningModule, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader
    ) -> None:
        if self._trainer is None:
            self.create_trainer()
        self._trainer.fit(model, train_dataloader, val_dataloader)
        return model
