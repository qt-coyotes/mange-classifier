import argparse

import torchvision.models as models
from torch import nn

from models.base import BaseModel


class DenseNetModel(BaseModel):
    def __init__(
        self,
        criterion: nn.Module,
        args: argparse.Namespace,
        architecture: int = 121,
    ):
        super().__init__(criterion, args)
        if architecture == 121:
            weights = models.DenseNet121_Weights.DEFAULT
            backbone = models.densenet121
        elif architecture == 161:
            weights = models.DenseNet161_Weights.DEFAULT
            backbone = models.densenet161
        elif architecture == 169:
            weights = models.DenseNet169_Weights.DEFAULT
            backbone = models.densenet169
        elif architecture == 201:
            weights = models.DenseNet201_Weights.DEFAULT
            backbone = models.densenet201
        else:
            raise ValueError(f"Unknown DenseNet architecture: {architecture}")
        backbone = backbone(
            weights=None if args.nonpretrained else weights
        )
        children = list(backbone.children())
        layers = children[:-1]
        self.image_backbone = nn.Sequential(*layers)
