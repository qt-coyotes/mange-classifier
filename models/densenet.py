import argparse

import torchvision.models as models
from torch import nn

from models.base import BaseModel


class DenseNetModel(BaseModel):
    def __init__(
        self,
        criterion: nn.Module,
        args: argparse.Namespace,
    ):
        super().__init__(criterion, args)
        if args.densenet_model == "DenseNet121":
            weights = models.DenseNet121_Weights.DEFAULT
            backbone = models.densenet121
        elif args.densenet_model == "DenseNet161":
            weights = models.DenseNet161_Weights.DEFAULT
            backbone = models.densenet161
        elif args.densenet_model == "DenseNet169":
            weights = models.DenseNet169_Weights.DEFAULT
            backbone = models.densenet169
        elif args.densenet_model == "DenseNet201":
            weights = models.DenseNet201_Weights.DEFAULT
            backbone = models.densenet201
        else:
            raise ValueError(f"Unknown DenseNet model: {args.densenet_model}")
        backbone = backbone(
            weights=None if args.nonpretrained else weights
        )
        children = list(backbone.children())
        layers = children[:-1]
        self.feature_extractor = nn.Sequential(*layers)
