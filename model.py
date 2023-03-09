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
    BinaryF1Score,
    BinaryFBetaScore,
    BinaryRecall,
    BinaryPrecision,
)

from metrics import BinaryExpectedCost
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)


class Module(LightningModule):
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        pretrained: bool,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.transforms = weights.transforms()
        backbone = models.resnet18(weights=weights if pretrained else None)
        self.graph_node_name = "fc"
        self.feature_extractor = create_feature_extractor(
            backbone, return_nodes=[self.graph_node_name]
        )
        children = list(backbone.children())
        layers = children[:-1]
        last_layer = children[-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(last_layer.in_features, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

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

    def forward(self, x: Tensor):
        x = self.transforms(x)
        representations = self.feature_extractor(x)
        x = self.flatten(representations)
        x = self.classifier(x)
        x = x.flatten()
        return x

    def step(self, batch: Tuple[Tensor], batch_idx: int, stage: str):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.float())
        self.log(f"{stage}_loss", loss, on_epoch=True)
        yhat = F.sigmoid(logits)
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
