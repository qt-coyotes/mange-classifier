import argparse
import csv
import gc
import json
import os
import time
from datetime import datetime, timedelta

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.accelerators import CUDAAccelerator
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn

from data import StratifiedGroupKFoldDataModule
from losses import BinaryExpectedCostLoss, BinaryMacroSoftFBetaLoss
from models.base import BaseModel
from models.densenet import DenseNetModel
from models.resnet import ResNetModel
from models.vit import ViTModel
from models.yolo import YoloModel


def main():
    models = {
        "DenseNet": DenseNetModel,
        "ResNet": ResNetModel,
        "ViT": ViTModel,
        "YOLO": YoloModel,
    }
    criterions = {
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
        "MacroSoftFBetaLoss": BinaryMacroSoftFBetaLoss(2),
        "ExpectedCostLoss": BinaryExpectedCostLoss(cfn=5),
    }
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    group = parser.add_argument_group("qt.coyote")
    group.add_argument(
        "--model",
        help="Which model to use",
        type=str,
        choices=list(models.keys()),
        default="ResNet",
    )
    group.add_argument(
        "--criterion",
        help="Which criterion to use",
        type=str,
        choices=list(criterions.keys()),
        default="ExpectedCostLoss",
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
        type=bool,
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
        action="store_true",
    )
    group.add_argument(
        "--nonpretrained",
        help="Do not use pretrained weights, train from scratch",
        action="store_true",
    )
    group.add_argument(
        "--compile",
        help="Compile the model",
        action="store_true",
    )
    group.add_argument(
        "--no_early_stopping",
        help="Disable early stopping",
        action="store_true",
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
        default="yolov8n-cls.pt",
    )
    args = parser.parse_args()
    if args.accelerator is None:
        args.accelerator = "auto"
    print(args)
    torch.backends.cudnn.deterministic = not args.nondeterministic
    Model = models[args.model]
    criterion = criterions[args.criterion]
    cross_validate(Model, criterion, args)
    # TODO: train final model


def cross_validate(Model: BaseModel, criterion: nn.Module, args: argparse.Namespace):
    # cross validation
    start_time = time.perf_counter()
    test_metrics = []
    datamodule = StratifiedGroupKFoldDataModule(args)
    for datamodule_i in datamodule:
        seed_everything(args.random_state, workers=True)
        callbacks = []
        if not args.no_early_stopping:
            callbacks.append(
                EarlyStopping("val_loss", patience=args.patience, mode="min")
            )
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
        )
        model = Model(criterion, args)
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
            test_metric = trainer.test(model, dataloaders=datamodule)
        else:
            test_metric = trainer.test(ckpt_path="best", dataloaders=datamodule)
        test_metrics.append(test_metric)
        if args.fast_dev_run:
            break
        del trainer
        del model
        gc.collect()

    end_time = time.perf_counter()
    time_elapsed = timedelta(seconds=end_time - start_time)
    save_logs(test_metrics, time_elapsed, args)


def save_logs(test_metrics, time_elapsed: timedelta, args: argparse.Namespace):
    cv_metrics = {"metric_confusion_matrix": []}
    for test_metric in test_metrics:
        test_metric = test_metric[0]
        test_metric_metric = test_metric["test_metric"]
        for key, value in test_metric_metric.items():
            key = key.replace("Binary", "")
            cv_metrics[key] = cv_metrics.get(key, 0) + value.item()
        cv_metrics["metric_confusion_matrix"].append(
            [
                [
                    int(test_metric["test_confusion_matrix_tn"]),
                    int(test_metric["test_confusion_matrix_fp"]),
                ],
                [
                    int(test_metric["test_confusion_matrix_fn"]),
                    int(test_metric["test_confusion_matrix_tp"]),
                ],
            ]
        )
        cv_metrics["loss"] = (
            cv_metrics.get("loss", 0) + test_metric["test_loss"]
        )

    for metric in cv_metrics:
        if isinstance(cv_metrics[metric], list):
            continue
        cv_metrics[metric] /= args.k

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logs = {
        "args": vars(args),
        "cv_metrics": cv_metrics,
        "time_elapsed": str(time_elapsed),
    }
    print(logs)
    if args.fast_dev_run:
        return
    with open(f"logs_{timestamp}.json", "w") as f:
        json.dump(logs, f, indent=4)

    with open(f"logs_{timestamp}.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([not logs["args"]["nonpretrained"]])
        patience = []
        max_epochs = []
        if logs["args"]["no_early_stopping"]:
            max_epochs.append(logs["args"]["max_epochs"])
        else:
            patience.append(logs["args"]["patience"])
        writer.writerow(patience)
        writer.writerow(max_epochs)
        writer.writerow([logs["args"]["batch_size"]])
        writer.writerow([logs["args"]["learning_rate"]])
        writer.writerow([])
        writer.writerow([logs["cv_metrics"]["ExpectedCost50"]])
        writer.writerow([logs["cv_metrics"]["ExpectedCost10"]])
        writer.writerow([logs["cv_metrics"]["ExpectedCost5"]])
        writer.writerow([logs["cv_metrics"]["F2"]])
        writer.writerow([logs["cv_metrics"]["F1"]])
        writer.writerow([logs["cv_metrics"]["Recall"]])
        writer.writerow([logs["cv_metrics"]["Precision"]])
        writer.writerow([logs["cv_metrics"]["AveragePrecision"]])
        writer.writerow([logs["cv_metrics"]["Accuracy"]])
        writer.writerow([logs["cv_metrics"]["AUROC"]])
        writer.writerow([json.dumps(logs["cv_metrics"]["metric_confusion_matrix"])])
        writer.writerow([logs["cv_metrics"]["loss"]])
        writer.writerow([])
        writer.writerow([logs["time_elapsed"]])


if __name__ == "__main__":
    main()
