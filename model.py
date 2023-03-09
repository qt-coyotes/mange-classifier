from typing import Tuple

import torch.nn.functional as F
import torchvision.models as models
from lightning.pytorch import LightningModule
from torch import Tensor, nn, optim
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryFBetaScore,
)

from metrics import BinaryExpectedCost


class Module(LightningModule):
    def __init__(self, batch_size: int, learning_rate: float = 1e-3):
        super().__init__()
        # weights = models.ViT_B_16_Weights.DEFAULT
        # backbone = models.vit_b_16(weights=weights)
        weights = models.ResNet18_Weights.DEFAULT
        self.transforms = weights.transforms()

        backbone = models.resnet18(weights=weights)
        children = list(backbone.children())
        layers = children[:-1]
        last_layer = children[-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(last_layer.in_features, 1), nn.Sigmoid()
        )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.metrics = nn.ModuleDict(
            {
                stage: MetricCollection(
                    [
                        BinaryExpectedCost(),
                        BinaryAccuracy(),
                        BinaryF1Score(),
                        BinaryFBetaScore(beta=2.0),
                        # BinaryAUROC(),
                        # BinaryAveragePrecision(),
                    ]
                )
                for stage in ["train", "test", "val"]
            }
        )

    def forward(self, x: Tensor):
        self.feature_extractor.eval()
        x = self.transforms(x)
        representations = self.feature_extractor(x)
        x = self.flatten(representations)
        x = self.classifier(x)
        x = x.flatten()
        return x

    def step(self, batch: Tuple[Tensor], batch_idx: int, stage: str):
        x, y = batch
        yhat = self.forward(x)
        loss = F.binary_cross_entropy(yhat, y.float())
        self.log(f"{stage}_loss", loss, on_epoch=True)
        metric = self.metrics[stage](yhat, y)
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
