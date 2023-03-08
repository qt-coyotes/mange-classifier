
from sklearn.model_selection import StratifiedGroupKFold
from lightning.pytorch import LightningDataModule
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class COCOImageDataset(Dataset):
    def __init__(
            self,
            images,
            labels,
            data_path,
            transform=None,
            target_transform=None
    ):
        self.images = images
        self.labels = labels
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_path = self.data_path / image["file_name"]
        img = read_image(str(image_path))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class StratifiedGroupKFoldDataModule(LightningDataModule):
    def __init__(self, k: int, i: int, data_path: Path, metadata_path: Path, batch_size: int, num_workers: int):
        super().__init__()
        self.k = k
        self.i = i
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.state = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.stage = stage

        with open(self.metadata_path, "r") as f:
            coco = json.load(f)

        images = coco["images"]
        image_id_to_image = {image["id"]: image for image in images}

        no_mange_category_ids = [2, 5]
        mange_category_ids = [1, 4]

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
        for image in X:
            groups.append(image["location"])

        trainvaltest_sgkf = StratifiedGroupKFold(
            n_splits=self.k,
            shuffle=False,
            random_state=None
        )
        trainvaltest_splits = list(trainvaltest_sgkf.split(X, y, groups=groups))
        trainval_indexes, test_indexes = trainvaltest_splits[self.i]

        test_X = [X[i] for i in test_indexes]
        test_y = [y[i] for i in test_indexes]

        self.dataset_test = COCOImageDataset(
            test_X,
            test_y,
            self.data_path
        )

        X_trainval = [X[i] for i in trainval_indexes]
        y_trainval = [y[i] for i in trainval_indexes]
        groups_trainval = [groups[i] for i in trainval_indexes]

        trainval_sgkf = StratifiedGroupKFold(
            n_splits=self.k,
            shuffle=False,
            random_state=None
        )
        trainval_splits = list(trainval_sgkf.split(X_trainval, y_trainval, groups=groups_trainval))
        train_indexes, val_indexes = trainval_splits[0]

        train_X = [X_trainval[i] for i in train_indexes]
        train_y = [y_trainval[i] for i in train_indexes]
        val_X = [X_trainval[i] for i in val_indexes]
        val_y = [y_trainval[i] for i in val_indexes]

        self.dataset_train = COCOImageDataset(
            train_X,
            train_y,
            self.data_path
        )
        self.dataset_val = COCOImageDataset(
            val_X,
            val_y,
            self.data_path
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage):
        pass
