import argparse
from typing import Tuple

import torch
from torch import nn, Tensor

from models.base import BaseModel


class AllPositiveModel(BaseModel):
    def __init__(
        self,
        criterion: nn.Module,
        args: argparse.Namespace,
    ):
        super().__init__(criterion, args)

    def step(self, batch: Tuple[Tensor], batch_idx: int, stage: str):
        x, y = batch
        yhat = torch.ones(x[0].shape[0], device=self.device)
        self.test_EC5(yhat, y)
        self.log(f"{stage}_EC5", self.test_EC5, prog_bar=True)
        self.log(f"{stage}_loss", -1, prog_bar=True)
        self.metrics[f"{stage}_metric"].update(yhat, y)
        self.metrics[f"{stage}_confusion_matrix"].update(yhat, y)
