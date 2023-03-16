import argparse
import json
import gc
import os
from datetime import datetime
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.accelerators import CUDAAccelerator
from lightning.pytorch.callbacks import EarlyStopping

from data import StratifiedGroupKFoldDataModule
from models.base import BaseModel
from models.resnet import ResNetModel
from models.vit import ViTModel
from models.yolo import YoloModel


def main():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    group = parser.add_argument_group("qt.coyote")
    group.add_argument(
        "--model",
        help="Which model to use",
        type=str,
        choices=["ViT", "ResNet", "YOLO"],
        default='ResNet'
    )
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
        default="data",
    )
    group.add_argument(
        "--metadata_path",
        help="Path to COCO metadata file",
        type=str,
        default="data/qt-coyotes-merged.json",
    )
    group.add_argument(
        "--num_workers",
        help="Number of workers for dataloader",
        type=int,
        default=os.cpu_count() - 2,
    )
    group.add_argument(
        "--persistent_workers",
        help="If True, the data loader will not shutdown the worker processes "
        "after a dataset has been consumed once. This allows to maintain the "
        "workers Dataset instances alive.",
        type=int,
        default=True,
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
        "--nondeterministic",
        help="This flag sets the torch.backends.cudnn.deterministic flag to false",
        action='store_true',
    )
    group.add_argument(
        "--nonpretrained",
        help="Do not use pretrained weights, train from scratch",
        action='store_true',
    )
    group.add_argument(
        "--compile",
        help="Compile the model",
        action='store_true',
    )
    group.add_argument(
        "--patience",
        help="Number of checks with no improvement after which training will be stopped.",
        type=int,
        default=10,
    )
    group.add_argument(
        "--yolo_model",
        help="Yolo pretrained model",
        type=str,
        default="yolov8n-cls.pt"
    )
    args = parser.parse_args()
    if args.accelerator is None:
        args.accelerator = "auto"
    print(args)
    torch.backends.cudnn.deterministic = not args.nondeterministic
    models = {
        'ResNet': ResNetModel,
        'ViT': ViTModel,
        'YOLO': YoloModel
    }
    Model = models[args.model]
    cross_validate(Model, args)
    # TODO: train final model


def cross_validate(Model: BaseModel, args: argparse.Namespace):
    # cross validation
    test_metrics = []
    datamodule = StratifiedGroupKFoldDataModule(args)
    for datamodule_i in datamodule:
        seed_everything(args.random_state, workers=True)
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[
                EarlyStopping("val_loss", patience=args.patience, mode="min")
            ],
        )
        model = Model(args)
        if args.compile and isinstance(trainer.accelerator, CUDAAccelerator):
            model = torch.compile(model)

        if args.auto_scale_batch_size or args.auto_lr_find:
            trainer.tune(model, datamodule=datamodule_i)
            print("Automatically found batch size and learning rate")
            print("Replace --auto_scale_batch_size and --auto_lr_find with:")
            print(f"--batch_size {model.batch_size}")
            print(f"--learning_rate {model.learning_rate}")
            break
        trainer.fit(model=model, train_dataloaders=datamodule_i)
        if args.fast_dev_run:
            test_metric = trainer.test(
                model,
                dataloaders=datamodule
            )
        else:
            test_metric = trainer.test(
                ckpt_path="best",
                dataloaders=datamodule
            )
        test_metrics.append(test_metric)
        if args.fast_dev_run:
            break
        del trainer
        del model
        gc.collect()

    save_logs(test_metrics, args)


def save_logs(test_metrics, args: argparse.Namespace):
    cv_metrics = {}
    for test_metric in test_metrics:
        test_metric = test_metric[0]
        test_metric_metric = test_metric["test_metric"]
        for key, value in test_metric_metric.items():
            key = key.replace("Binary", "")
            cv_metrics[key] = cv_metrics.get(key, 0) + value.item()
        cv_metrics["loss"] = (
            cv_metrics.get("loss", 0) + test_metric["test_loss"]
        )

    for metric in cv_metrics:
        cv_metrics[metric] /= args.k

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logs = {
        "args": vars(args),
        "cv_metrics": cv_metrics,
    }
    print(logs)
    if args.fast_dev_run:
        return
    with open(f"logs_{timestamp}.json", "w") as f:
        json.dump(logs, f, indent=4)


if __name__ == "__main__":
    main()
