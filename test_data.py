import unittest

import itertools
import os
import argparse

from data import StratifiedGroupKFoldDataModule


class TestData(unittest.TestCase):
    def test_kfold(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", help="Batch size", type=int, default=1)
        parser.add_argument(
            "--k",
            help="Number of folds in k-fold cross validation",
            type=int,
            default=5,
        )
        parser.add_argument(
            "--data_path",
            help="Path to images",
            type=str,
            default="data",
        )
        parser.add_argument(
            "--metadata_path",
            help="Path to COCO metadata file",
            type=str,
            default="data/qt-coyotes-merged.json",
        )
        parser.add_argument(
            "--num_workers",
            help="Number of workers for dataloader",
            type=int,
            default=os.cpu_count() - 2,
        )
        parser.add_argument(
            "--persistent_workers",
            help="If True, the data loader will not shutdown the worker processes "
            "after a dataset has been consumed once. This allows to maintain the "
            "workers Dataset instances alive.",
            type=bool,
            default=True,
        )
        parser.add_argument(
            "--shuffle",
            help="Whether to shuffle each class's samples before splitting into "
            "batches. Note that the samples within each split will not be "
            "shuffled. This implementation can only shuffle groups that have "
            "approximately the same y distribution, no global shuffle will be "
            "performed.",
            type=bool,
            default=True,
        )
        parser.add_argument(
            "--random_state",
            help="When shuffle is True, random_state affects the ordering of the "
            "indices, which controls the randomness of each fold for each class. "
            "Otherwise, leave random_state as None. Pass an int for reproducible "
            "output across multiple function calls.",
            type=int,
            default=42,
        )
        args = parser.parse_known_args([])[0]
        datamodule = StratifiedGroupKFoldDataModule(args)
        datamodule.prepare_data()
        datamodule.setup(None)
        for i, datamodule_i in enumerate(datamodule):
            D = {}
            train = datamodule_i.train_dataloader()
            val = datamodule_i.val_dataloader()
            test = datamodule_i.test_dataloader()
            ltrain = len(train.dataset)
            lval = len(val.dataset)
            ltest = len(test.dataset)
            print(f"Fold {i}: train={ltrain}, val={lval}, test={ltest}, total={ltrain+lval+ltest}")
            for dataloader, stage in zip([train, val, test], ["train", "val", "test"]):
                for j, (X, y) in enumerate(dataloader):
                    h = hash((X, y))
                    if h not in D:
                        D[h] = (X, y)
                    else:
                        raise ValueError(f"Duplicate ({X}, {y}), [{D[h]}] found in {i}:{stage}:{j}")
