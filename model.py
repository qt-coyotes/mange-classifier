from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.models as models
from lightning.pytorch import LightningModule
from torch import Tensor, nn, optim
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryFBetaScore,
)

from metrics import BinaryExpectedCost


class Module(LightningModule):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        layers = list(backbone.children())[:-1]
        self.transforms = weights.transforms()
        self.feature_extractor = nn.Sequential(*layers)
        num_filters = backbone.fc.in_features
        self.classifier = nn.Sequential(nn.Linear(num_filters, 1), nn.Sigmoid())
        self.metrics = MetricCollection(
            [
                BinaryExpectedCost(),
                BinaryAccuracy(),
                BinaryF1Score(),
                BinaryFBetaScore(beta=2.0),
                BinaryAUROC(),
                BinaryAveragePrecision(),
            ]
        )

    def forward(self, x: Tensor):
        self.feature_extractor.eval()
        x = self.transforms(x)
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        x = x.flatten()
        return x

    def training_step(self, batch: Tuple[Tensor], batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = F.binary_cross_entropy(yhat, y.float())
        self.log("train_loss", loss)
        metric = self.metrics(yhat, y)
        self.log("train_metric", metric)
        return loss

    def validation_step(self, batch: Tuple[Tensor], batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = F.binary_cross_entropy(yhat, y.float())
        self.log("val_loss", loss)
        metric = self.metrics(yhat, y)
        self.log("val_metric", metric)
        return loss

    def test_step(self, batch: Tuple[Tensor], batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = F.binary_cross_entropy(yhat, y.float())
        self.log("test_loss", loss)
        metric = self.metrics(yhat, y)
        self.log("test_metric", metric)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
