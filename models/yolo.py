import argparse

from ultralytics import YOLO
from models.base import BaseModel
from torch import nn


class YoloModel(BaseModel):
    def __init__(
        self,
        args: argparse.Namespace,
    ):
        super().__init__(args)
        model = YOLO(args.yolo_model).model
        list(
            list(model.children())[0].children()
        )[-1].linear.out_features = 1
        self.feature_extractor = model.model
        self.classifier = nn.Identity()
