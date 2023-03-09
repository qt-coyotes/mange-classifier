import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping

from data import StratifiedGroupKFoldDataModule
from model import Module


def main():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    group = parser.add_argument_group("qt.coyote")
    group.add_argument("--batch_size", help="Batch size", type=int, default=32)
    group.add_argument(
        "--learning_rate", help="Learning rate", type=float, default=1e-3
    )
    group.add_argument(
        "--k",
        help="Number of folds in k-fold cross validation",
        type=int,
        default=5,
    )
    group.add_argument(
        "--data_path",
        help="Path to images",
        type=str,
        default="data/CHIL/images",
    )
    group.add_argument(
        "--metadata_path",
        help="Path to COCO metadata file",
        type=str,
        default="data/CHIL/CHIL_uwin_mange_Marit_07242020.json",
    )
    group.add_argument(
        "--num_workers",
        help="Number of workers for dataloader",
        type=int,
        default=16,
    )
    group.add_argument(
        "--shuffle",
        help="Whether to shuffle each class's samples before splitting into "
        "batches. Note that the samples within each split will not be "
        "shuffled. This implementation can only shuffle groups that have "
        "approximately the same y distribution, no global shuffle will be "
        "performed.",
        type=bool,
        default=True,
    )
    group.add_argument(
        "--random_state",
        help="When shuffle is True, random_state affects the ordering of the "
        "indices, which controls the randomness of each fold for each class. "
        "Otherwise, leave random_state as None. Pass an int for reproducible "
        "output across multiple function calls.",
        type=int,
        default=42,
    )
    group.add_argument(
        "--deterministic",
        help="This flag sets the torch.backends.cudnn.deterministic flag.",
        type=bool,
        default=True,
    )
    group.add_argument(
        "--patience",
        help="Number of checks with no improvement after which training will be stopped.",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = args.deterministic
    if args.accelerator is None:
        args.accelerator = "auto"
    run(args)


def run(args):
    # cross validation
    test_metrics = []
    for i in range(args.k):
        seed_everything(args.random_state, workers=True)
        model = Module(
            batch_size=args.batch_size, learning_rate=args.learning_rate
        )
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[
                EarlyStopping("val_loss", patience=args.patience, mode="min")
            ],
        )
        trainer.logger._log_graph = True
        datamodule = StratifiedGroupKFoldDataModule(
            args.k,
            i,
            Path(args.data_path),
            Path(args.metadata_path),
            args.batch_size,
            args.num_workers,
            args.shuffle,
            args.random_state,
        )
        trainer.tune(model, datamodule=datamodule)
        if args.auto_scale_batch_size or args.auto_lr_find:
            print("Automatically found batch size and learning rate")
            print("Replace --auto_scale_batch_size and --auto_lr_find with:")
            print(f"--batch_size {model.batch_size}")
            print(f"--learning_rate {model.learning_rate}")
            break
        trainer.fit(model=model, train_dataloaders=datamodule)
        test_metric = trainer.test(ckpt_path="best", dataloaders=datamodule)
        test_metrics.append(test_metric)

    k = 5

    cv_metrics = {}
    for test_metric in test_metrics:
        test_metric = test_metric[0]
        cv_metrics["loss"] = (
            cv_metrics.get("loss", 0) + test_metric["test_loss"]
        )
        test_metric = test_metric["test_metric"]
        for key, value in test_metric.items():
            key = key.replace("Binary", "")
            cv_metrics[key] = cv_metrics.get(key, 0) + value.item()

    for metric in cv_metrics:
        cv_metrics[metric] /= k

    timestamp = datetime.now().strftime("%Y%-m-%d-%H-%M-%S")
    with open(f"cv_metrics_{timestamp}.json", "w") as f:
        json.dump(cv_metrics, f, indent=4)

    # TODO: train final model


if __name__ == "__main__":
    main()
