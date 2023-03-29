import argparse
from typing import Tuple

from ..transforms import CoyoteCrop

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn, optim
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryFBetaScore,
    BinaryPrecision,
    BinaryRecall,
    BinaryConfusionMatrix,
)

from metrics import BinaryExpectedCost


class BaseModel(LightningModule):
    def __init__(self, criterion: nn.Module, args: argparse.Namespace):
        super().__init__()
        self.transforms = nn.Identity()
        self.feature_extractor = NotImplementedError()
        self.flatten = nn.Flatten()
        self.classifier = nn.LazyLinear(1)
        self.criterion = criterion
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_factor = args.scheduler_factor
        self.crop_coyote = CoyoteCrop() if args.crop_coyote else nn.Identity()
        self.return_node = None
        metrics = {}
        for stage in ["train", "test", "val"]:
            metrics[f"{stage}_confusion_matrix"] = BinaryConfusionMatrix()
            metrics[f"{stage}_metric"] = MetricCollection(
                {
                    "ExpectedCost5": BinaryExpectedCost(),
                    "ExpectedCost10": BinaryExpectedCost(cfn=10.),
                    "ExpectedCost50": BinaryExpectedCost(cfn=50.),
                    "F2": BinaryFBetaScore(beta=2.0),
                    "F1": BinaryF1Score(),
                    "Recall": BinaryRecall(),
                    "Precision": BinaryPrecision(),
                    "AveragePrecision": BinaryAveragePrecision(),
                    "Accuracy": BinaryAccuracy(),
                    "AUROC": BinaryAUROC(),
                }
            )
        self.metrics = nn.ModuleDict(metrics)

    def y(self, x: Tensor):
        x = x.float()
        x = self.crop_coyote(x)
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
        yhat = torch.sigmoid(logits)
        loss = self.criterion(yhat, y.float())
        self.log(f"{stage}_loss", loss, on_epoch=True)
        self.metrics[f"{stage}_metric"].update(yhat, y)
        self.metrics[f"{stage}_confusion_matrix"].update(yhat, y)
        return loss

    def training_step(self, batch: Tuple[Tensor], batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: Tuple[Tensor], batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch: Tuple[Tensor], batch_idx):
        return self.step(batch, batch_idx, "test")

    def epoch_end(self, outputs, stage: str):
        metrics = self.metrics[f"{stage}_metric"].compute()
        self.log(f"{stage}_metric", metrics)
        confmat = self.metrics[f"{stage}_confusion_matrix"].compute().float()
        (tn, fp), (fn, tp) = confmat
        self.log(f"{stage}_confusion_matrix_tn", tn, reduce_fx=torch.sum)
        self.log(f"{stage}_confusion_matrix_fp", fp, reduce_fx=torch.sum)
        self.log(f"{stage}_confusion_matrix_fn", fn, reduce_fx=torch.sum)
        self.log(f"{stage}_confusion_matrix_tp", tp, reduce_fx=torch.sum)

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
