import argparse
from typing import Tuple

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn, optim
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryFBetaScore, BinaryPrecision, BinaryRecall

from metrics import BinaryExpectedCost


class BaseModel(LightningModule):
    def __init__(
        self,
        args: argparse.Namespace
    ):
        super().__init__()
        self.transforms = nn.Identity()
        self.feature_extractor = NotImplementedError()
        self.flatten = nn.Flatten()
        self.classifier = nn.LazyLinear(1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.return_node = None

        self.metrics = nn.ModuleDict(
            {
                f"{stage}_metric": MetricCollection(
                    [
                        BinaryExpectedCost(),
                        BinaryFBetaScore(beta=2.0),
                        BinaryF1Score(),
                        BinaryRecall(),
                        BinaryPrecision(),
                        BinaryAccuracy(),
                        # BinaryAUROC(),
                        # BinaryAveragePrecision(),
                    ]
                )
                for stage in ["train", "test", "val"]
            }
        )

    def y(self, x: Tensor):
        x = x.float()
        x = self.transforms(x)
        x = self.feature_extractor(x)
        if self.return_node:
            x = x[self.return_node]
        x = self.flatten(x)
        x = self.classifier(x)
        y = x.flatten()
        return y

    def step(self, batch: Tuple[Tensor], batch_idx: int, stage: str):
        x, y = batch
        logits = self.y(x)
        loss = self.criterion(logits, y.float())
        self.log(f"{stage}_loss", loss, on_epoch=True)
        yhat = torch.sigmoid(logits)
        metric = self.metrics[f"{stage}_metric"](yhat, y)
        self.log(f"{stage}_metric", metric)
        return loss

    def training_step(self, batch: Tuple[Tensor], batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: Tuple[Tensor], batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch: Tuple[Tensor], batch_idx):
        return self.step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
