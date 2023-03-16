import argparse

import torchvision.models as models
from torch import nn

from models.base import BaseModel


class DenseNetModel(BaseModel):
    def __init__(
        self,
        args: argparse.Namespace,
    ):
        super().__init__(args)
        weights = models.DenseNet121_Weights.DEFAULT
        self.transforms = weights.transforms()
        backbone = models.densenet121(
            weights=None if args.nonpretrained else weights
        )
        children = list(backbone.children())
        layers = children[:-1]
        self.feature_extractor = nn.Sequential(*layers)
