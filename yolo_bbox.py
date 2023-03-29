from ultralytics import YOLO
import torch
import torchvision
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import random

import numpy as np
from PIL import Image
import json
from math import inf
from pathlib import Path

from data import COCOImageDataset


SAVE = False # If set to True saves images with bounding boxes.
BATCH_SIZE = 50 # Set number of photos to be passed into Yolo model at one time.
NUM_PHOTOS = 200 # Total number of photos to process. Can set > than total photos in dataset to process all photos.


class BBOXCOCOImageDataset(COCOImageDataset):
    def __getitem__(self, idx):
        image = self.images[idx]
        image_path = self.data_path / image["file_name"]
        img = Image.open(str(image_path))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


def main():

    with open(Path('data/qt-coyotes-merged.json'), "r") as f:
        coco = json.load(f)

    images = coco["images"]

    image_id_to_image = {image["id"]: image for image in images}

    mange_category_ids = {1}
    no_mange_category_ids = {2}

    min_image_shape = None
    min_image_area = inf
    for image in images:
        width = image["width"]
        height = image["height"]
        area = width * height
        if area < min_image_area:
            min_image_shape = (height, width)
            min_image_area = area

    equal_size_transform = torchvision.transforms.Resize(
        min_image_shape
    )


    no_mange_annotations = []
    mange_annotations = []
    for annotation in coco["annotations"]:
        if annotation["category_id"] in no_mange_category_ids:
            no_mange_annotations.append(annotation)
        elif annotation["category_id"] in mange_category_ids:
            mange_annotations.append(annotation)

    X = []
    for annotation in no_mange_annotations + mange_annotations:
        image_id = annotation["image_id"]
        X.append(image_id_to_image[image_id])

    y = [0] * len(no_mange_annotations) + [1] * len(mange_annotations)

    data = BBOXCOCOImageDataset(X, y, Path("data"), equal_size_transform)

    index_list = [x for x in range(len(data))]
    random.shuffle(index_list)
    i = 0

    while NUM_PHOTOS - (BATCH_SIZE * i) and index_list:
        batch_indexes = index_list[:BATCH_SIZE]
        index_list = index_list[BATCH_SIZE:]
        i+=1

        coyote_image_list = [data[i][0] for i in batch_indexes]
        model = YOLO('yolov8x.pt')
        outputs = model(coyote_image_list, save=SAVE)
                

if __name__ =='__main__':
    main()