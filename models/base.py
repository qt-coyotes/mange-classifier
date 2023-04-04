import argparse
from typing import Tuple

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
import torchvision.transforms as transforms
from metrics import BinaryExpectedCost
from losses import HybridLoss


class BaseModel(LightningModule):
    def __init__(self, criterion: nn.Module, args: argparse.Namespace):
        super().__init__()
        self.transforms = nn.Identity()
        self.image_backbone = NotImplementedError()
        self.flatten = nn.Flatten()
        # self.tabular_backbone = nn.Sequential(
        #     nn.LazyBatchNorm1d(),
        #     nn.LazyLinear(args.tabular_hidden_size),
        #     nn.ReLU(),
        #     nn.LazyBatchNorm1d(),
        #     nn.Linear(args.tabular_hidden_size, args.tabular_hidden_size),
        #     nn.ReLU(),
        # )
        self.tabular_backbone = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(args.tabular_hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(1)
        )
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # transforms.RandomPerspective(),
            transforms.GaussianBlur(3, sigma=(0.1, 2)),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
        ])
        self.criterion = criterion
        self.batch_size = args.batch_size
        self.no_data_augmentation = args.no_data_augmentation
        self.learning_rate = args.learning_rate
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_factor = args.scheduler_factor
        self.return_node = None
        metrics = {}
        for stage in ["train", "test", "val"]:
            metrics[f"{stage}_confusion_matrix"] = BinaryConfusionMatrix()
            metrics[f"{stage}_metric"] = MetricCollection(
                {
                    "ExpectedCost5": BinaryExpectedCost(cfn=5.),
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
        self.train_EC5 = BinaryExpectedCost(cfn=5.)
        self.val_EC5 = BinaryExpectedCost(cfn=5.)
        self.test_EC5 = BinaryExpectedCost(cfn=5.)
        self.metrics = nn.ModuleDict(metrics)

    def y(self, x: Tuple[Tensor, Tensor]):
        i, t = x
        i = i.float()
        if not self.no_data_augmentation:
            i = self.augmentations(i)
        i = self.transforms(i)
        i = self.image_backbone(i)
        if self.return_node:
            i = i[self.return_node]
        i = self.flatten(i)
        t = self.tabular_backbone(t)
        x = torch.cat((i, t), dim=1)
        x = self.classifier(x)
        y = x.flatten()
        return y

    def step(self, batch: Tuple[Tensor], batch_idx: int, stage: str):
        x, y = batch
        logits = self.y(x)
        yhat = torch.sigmoid(logits)
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            loss = self.criterion(logits, y.float())
        elif isinstance(self.criterion, HybridLoss):
            loss = self.criterion(logits, y.float(), batch_idx)
        else:
            loss = self.criterion(yhat, y.float())
        self.log(f"{stage}_loss", loss)
        if stage == "train":
            self.train_EC5(yhat, y)
            self.log(f"{stage}_EC5", self.train_EC5, prog_bar=True)
        elif stage == "val":
            self.val_EC5(yhat, y)
            self.log(f"{stage}_EC5", self.val_EC5, prog_bar=True)
        elif stage == "test":
            self.test_EC5(yhat, y)
            self.log(f"{stage}_EC5", self.test_EC5, prog_bar=True)
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
