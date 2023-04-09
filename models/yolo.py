import argparse

from torch import nn
from ultralytics import YOLO

from models.base import BaseModel


class YoloModel(BaseModel):
    def __init__(
        self,
        criterion: nn.Module,
        args: argparse.Namespace,
        architecture: str = "yolov5n-cls.pt",
    ):
        super().__init__(criterion, args)
        model = YOLO(architecture).model
        list(list(model.children())[0].children())[-1].linear = nn.Identity()
        for k, v in model.named_parameters():
            v.requires_grad = True
        self.image_backbone = model.model
