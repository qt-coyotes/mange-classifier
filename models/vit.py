import argparse

from torch import nn
import torchvision.models as models
from torchvision.models.feature_extraction import (
    create_feature_extractor,
)  # , get_graph_node_names

from models.base import BaseModel


class ViTModel(BaseModel):
    def __init__(
        self,
        criterion: nn.Module,
        args: argparse.Namespace,
    ):
        super().__init__(criterion, args)
        weights = models.ViT_B_16_Weights.DEFAULT
        self.transforms = weights.transforms()
        backbone = models.vit_b_16(weights=weights)
        self.return_node = "getitem_5"
        self.feature_extractor = create_feature_extractor(
            backbone, return_nodes=[self.return_node]
        )
