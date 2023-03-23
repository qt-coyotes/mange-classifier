import argparse

from ultralytics import YOLO
from models.base import BaseModel
from torch import nn


class YoloModel(BaseModel):
    def __init__(
        self,
        criterion: nn.Module,
        args: argparse.Namespace,
    ):
        super().__init__(criterion, args)
        model = YOLO(args.yolo_model).model
        list(list(model.children())[0].children())[-1].linear = nn.Identity()
        for k, v in model.named_parameters():
            v.requires_grad = True
        self.feature_extractor = model.model
