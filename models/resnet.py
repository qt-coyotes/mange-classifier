import argparse

import torchvision.models as models
from torch import nn

from models.base import BaseModel


class ResNetModel(BaseModel):
    def __init__(
        self,
        criterion: nn.Module,
        args: argparse.Namespace,
        architecture: int = 18,
    ):
        super().__init__(criterion, args)
        if architecture == 18:
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18
        elif architecture == 34:
            weights = models.ResNet34_Weights.DEFAULT
            backbone = models.resnet34
        elif architecture == 50:
            weights = models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50
        elif architecture == 101:
            weights = models.ResNet101_Weights.DEFAULT
            backbone = models.resnet101
        elif architecture == 152:
            weights = models.ResNet152_Weights.DEFAULT
            backbone = models.resnet152
        else:
            raise ValueError(f"Unknown ResNet model: {architecture}")
        backbone = backbone(
            weights=None if args.nonpretrained else weights
        )
        self.transforms = weights.transforms()
        children = list(backbone.children())
        layers = children[:-1]
        self.image_backbone = nn.Sequential(*layers)
