import argparse

import torchvision.models as models
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor  # , get_graph_node_names

from models.base import BaseModel


class ViTModel(BaseModel):
    def __init__(
        self,
        criterion: nn.Module,
        args: argparse.Namespace,
        architecture: str = "B/16",
    ):
        super().__init__(criterion, args)
        if architecture == "B/16":
            weights = models.ViT_B_16_Weights.DEFAULT
            backbone = models.vit_b_16(weights=weights)
        elif architecture == "B/32":
            weights = models.ViT_B_32_Weights.DEFAULT
            backbone = models.vit_b_32(weights=weights)
        elif architecture == "L/16":
            weights = models.ViT_L_16_Weights.DEFAULT
            backbone = models.vit_l_16(weights=weights)
        elif architecture == "L/32":
            weights = models.ViT_L_32_Weights.DEFAULT
            backbone = models.vit_l_32(weights=weights)
        elif architecture == "H/14":
            weights = models.ViT_H_14_Weights.DEFAULT
            backbone = models.vit_h_14(weights=weights)
        else:
            raise ValueError(f"Unknown ViT architecture: {architecture}")
        self.transforms = weights.transforms()
        self.return_node = "getitem_5"
        self.image_backbone = create_feature_extractor(
            backbone, return_nodes=[self.return_node]
        )
