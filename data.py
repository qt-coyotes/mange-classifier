import argparse
import json
from pathlib import Path

import torch
import numpy as np
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.io import read_image

from transforms import SquarePad


class COCOImageDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        data_path,
        args: argparse.Namespace,
        transform=None,
        target_transform=None,
        pos_weight=None,
    ):
        self.images = images
        self.labels = labels
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.args = args
        self.pos_weight = pos_weight
        years = np.array([image["year"] for image in images])
        self.year_mean = years.mean()
        self.year_std = years.std()
        months = np.array([image["month"] for image in images])
        self.month_mean = months.mean()
        self.month_std = months.std()
        days = np.array([image["day"] for image in images])
        self.day_mean = days.mean()
        self.day_std = days.std()
        hours = np.array([image["hour"] for image in images])
        self.hour_mean = hours.mean()
        self.hour_std = hours.std()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.args.no_crop:
            image_path = self.data_path / image["file_name"]
            img = read_image(str(image_path))
        else:
            try:
                cropped_image_path = (
                    self.data_path / "megadetected" / image["file_name"]
                )
                img = read_image(str(cropped_image_path))
            except Exception:
                image_path = self.data_path / image["file_name"]
                img = read_image(str(image_path))
        if self.args.no_tabular_features:
            tabular = torch.tensor([], dtype=torch.float32)
        else:
            tabular = torch.tensor([
                image["is_color"],
                (image["year"] - self.year_mean) / self.year_std,
                (image["month"] - self.month_mean) / self.month_std,
                # image["day"],
                (image["hour"] - self.hour_mean) / self.hour_std,
                # image["minute"],
                image["latitude"] / 90.,
                image["longitude"] / 180.,
            ], dtype=torch.float32)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return (img, tabular), label


class StratifiedGroupKFoldDataModule(LightningDataModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.i = -1
        self.data_path = Path(args.data_path)
        self.metadata_path = Path(args.metadata_path)
        self.args = args
        self.dataset_train = []
        self.dataset_val = []
        self.dataset_test = []

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.stage = stage

        with open(self.metadata_path, "r") as f:
            coco = json.load(f)

        images = coco["images"]
        image_id_to_image = {image["id"]: image for image in images}

        mange_category_ids = {1}
        no_mange_category_ids = {2}

        equal_size_transform = T.Compose(
            [
                SquarePad(),
                T.Resize((self.args.crop_size, self.args.crop_size), antialias=True),
            ]
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

        groups = []
        if not self.args.no_external_group:
            for image in X:
                groups.append(image["location"])
            trainvaltest_sgkf = StratifiedGroupKFold(
                n_splits=self.args.k,
                shuffle=self.args.shuffle,
                random_state=self.args.random_state,
            )
            trainvaltest_splits = list(trainvaltest_sgkf.split(X, y, groups=groups))
        else:
            print("WARNING: No external grouping!")
            trainvaltest_skf = StratifiedKFold(
                n_splits=self.args.k,
                shuffle=self.args.shuffle,
                random_state=self.args.random_state,
            )
            trainvaltest_splits = list(trainvaltest_skf.split(X, y))

        for i in range(self.args.k):
            trainval_indexes, test_indexes = trainvaltest_splits[i]

            test_X = [X[i] for i in test_indexes]
            test_y = [y[i] for i in test_indexes]

            self.dataset_test.append(
                COCOImageDataset(
                    test_X,
                    test_y,
                    self.data_path,
                    self.args,
                    equal_size_transform,
                )
            )

            X_trainval = [X[i] for i in trainval_indexes]
            y_trainval = [y[i] for i in trainval_indexes]

            if self.args.internal_group:
                groups_trainval = [groups[i] for i in trainval_indexes]
                trainval_sgkf = StratifiedGroupKFold(
                    n_splits=self.args.internal_k,
                    shuffle=False,
                    random_state=None,
                )
                trainval_splits = list(
                    trainval_sgkf.split(
                        X_trainval, y_trainval, groups=groups_trainval
                    )
                )
            else:
                trainval_skf = StratifiedKFold(
                    n_splits=self.args.internal_k,
                    shuffle=False,
                    random_state=None,
                )
                trainval_splits = list(
                    trainval_skf.split(X_trainval, y_trainval)
                )
            train_indexes, val_indexes = trainval_splits[0]

            train_X = [X_trainval[i] for i in train_indexes]
            train_y = [y_trainval[i] for i in train_indexes]
            val_X = [X_trainval[i] for i in val_indexes]
            val_y = [y_trainval[i] for i in val_indexes]

            n1 = sum(train_y)
            n0 = len(train_y) - n1
            p = n0 / n1

            self.dataset_train.append(
                COCOImageDataset(
                    train_X,
                    train_y,
                    self.data_path,
                    self.args,
                    equal_size_transform,
                    pos_weight=p,
                )
            )
            self.dataset_val.append(
                COCOImageDataset(
                    val_X,
                    val_y,
                    self.data_path,
                    self.args,
                    equal_size_transform,
                )
            )

    def train_dataset(self):
        return self.dataset_train[self.i]

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train[self.i],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val[self.i],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test[self.i],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
        )

    def teardown(self, stage):
        pass

    def __iter__(self):
        self.i = -1
        return self

    def __next__(self):
        self.i += 1
        if self.i < self.args.k:
            return self
        else:
            raise StopIteration
