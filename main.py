import argparse
import gc
import os
import time
from datetime import timedelta

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.accelerators import CUDAAccelerator
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn

from data import StratifiedGroupKFoldDataModule
from logs import (
    aggregate_logs,
    generate_logs,
    log_to_gsheet,
    log_to_json,
    extract_lightning_logs,
)
from losses import (
    BinaryExpectedCostLoss,
    BinaryMacroSoftFBetaLoss,
    BinarySurrogateFBetaLoss,
    HybridLoss,
)
from models.all_negative import AllNegativeModel
from models.all_positive import AllPositiveModel
from models.base import BaseModel
from models.densenet import DenseNetModel
from models.random import RandomModel
from models.resnet import ResNetModel
from models.vit import ViTModel
from models.yolo import YoloModel

NO_TRAIN_MODELS = {"Random", "AllPositive", "AllNegative"}


def main():
    models = {
        "DenseNet": DenseNetModel,
        "ResNet": ResNetModel,
        "ViT": ViTModel,
        "YOLO": YoloModel,
        "Random": RandomModel,
        "AllPositive": AllPositiveModel,
        "AllNegative": AllNegativeModel,
        "Random": RandomModel,
    }
    criterions_set = {
        "BCELoss",
        "wBCELoss",
        "awBCELoss",
        "MacroSoftFBetaLoss",
        "ExpectedCostLoss",
        "SurrogateFBetaLoss",
        "HybridLoss",
        "HybridLoss",
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
        choices=criterions_set,
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
        "--internal_k",
        help="Number of folds for train/test split",
        type=int,
        default=5,
    )
    group.add_argument(
        "--no_external_group",
        help="Use grouped k-fold cross validation in external k-fold cross validation",
        action="store_true",
    )
    group.add_argument(
        "--internal_group",
        help="Use grouped k-fold cross validation in internal k-fold cross validation",
        action="store_true",
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
        "--no_crop",
        help="Disable cropping",
        action="store_true",
    )
    group.add_argument(
        "--no_data_augmentation",
        help="Disable data augmentation",
        action="store_true",
    )
    group.add_argument(
        "--patience",
        help="Number of checks with no improvement after which training will be stopped.",
        type=int,
        default=5,
    )
    group.add_argument(
        "--scheduler_factor",
        help="Factor by which the lr will be decreased",
        type=float,
        default=0.5,
    )
    group.add_argument(
        "--scheduler_patience",
        help="Number of checks with no improvement after which lr will decrease",
        type=int,
        default=4,
    )
    group.add_argument(
        "--yolo_model",
        help="Yolo pretrained model",
        type=str,
        default="yolov8n-cls.pt",
    )
    group.add_argument(
        "--resnet_model",
        help="ResNet model",
        type=str,
        default="ResNet18",
    )
    group.add_argument(
        "--densenet_model",
        help="DenseNet model",
        type=str,
        default="DenseNet121",
    )
    group.add_argument(
        "--vit_model",
        help="DenseNet model",
        type=str,
        default="ViT-B_16",
    )
    group.add_argument(
        "--crop_size",
        help="Crop size",
        type=int,
        default=224,
    )
    group.add_argument(
        "--criterion_pos_weight",
        help="Weight for positive class for BCEWithLogitsLoss",
        type=float,
        default=10.0,
    )
    group.add_argument(
        "--criterion_beta", help="Beta for F-beta loss", type=float, default=5.0
    )
    group.add_argument(
        "--criterion_cfn",
        help="Cost false negative for ExpectedCostLoss",
        type=float,
        default=5.0,
    )
    group.add_argument(
        "--dropout_p",
        help="Dropout probability",
        type=float,
        default=0.2,
    )
    group.add_argument(
        "--tabular_hidden_size",
        help="Size of the tabular hidden layers",
        type=int,
        default=32,
    )
    group.add_argument(
        "--no_tabular_features",
        help="Do not use tabular features",
        action="store_true",
    )
    group.add_argument(
        "--monitor",
        help="Metric to monitor",
        type=str,
        default="val_EC5",
    )
    group.add_argument(
        "--no_save_checkpoint",
        action="store_true",
        help="Backup the checkpoint",
    )
    group.add_argument("--message", help="Message to log", type=str)
    args = parser.parse_args()
    if args.accelerator is None:
        args.accelerator = "auto"
    print(args)
    torch.backends.cudnn.deterministic = not args.nondeterministic
    Model = models[args.model]
    criterions = {
        "BCELoss": nn.BCEWithLogitsLoss(),
        "wBCELoss": nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(args.criterion_pos_weight)
        ),
        "awBCELoss": "awBCELoss",
        "MacroSoftFBetaLoss": BinaryMacroSoftFBetaLoss(args.criterion_beta),
        "ExpectedCostLoss": BinaryExpectedCostLoss(cfn=args.criterion_cfn),
        "SurrogateFBetaLoss": BinarySurrogateFBetaLoss(args.criterion_beta),
        "HybridLoss": "HybridLoss",
    }
    criterion = criterions[args.criterion]
    cross_validate(Model, criterion, args)
    # TODO: train final model


def cross_validate(
    Model: BaseModel, criterion: nn.Module, args: argparse.Namespace
):
    # cross validation
    start_time = time.perf_counter()
    test_metrics = []
    datamodule = StratifiedGroupKFoldDataModule(args)
    for datamodule_i in datamodule:
        seed_everything(args.random_state, workers=True)
        callbacks = []
        if not args.no_early_stopping:
            callbacks.append(
                EarlyStopping(args.monitor, patience=args.patience, mode="min")
            )
            model_checkpoint = ModelCheckpoint(
                monitor=args.monitor,
            )
            callbacks.append(model_checkpoint)
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            log_every_n_steps=1000,
        )
        if criterion == "awBCELoss" or criterion == "HybridLoss":
            datamodule_i.setup(None)
            p = datamodule_i.train_dataset().pos_weight
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(p))
        if criterion == "HybridLoss":
            criterion = HybridLoss(
                criterion, BinaryExpectedCostLoss(cfn=args.criterion_cfn)
            )
        if criterion == "HybridLoss":
            criterion = HybridLoss(
                criterion, BinaryExpectedCostLoss(cfn=args.criterion_cfn)
            )
        model = Model(criterion, args)
        if args.compile and isinstance(trainer.accelerator, CUDAAccelerator):
            model = torch.compile(model)

        if args.auto_scale_batch_size or args.auto_lr_find:
            datamodule_i.setup(None)
            X, _ = next(iter(datamodule_i.train_dataloader()))
            _ = model(X)
            datamodule_i.setup(None)
            X, _ = next(iter(datamodule_i.train_dataloader()))
            _ = model(X)
            trainer.tune(model, datamodule=datamodule_i)
            if args.auto_lr_find:
                print(
                    f"Automatically found learning rate: {model.learning_rate}"
                )
                args.learning_rate = model.learning_rate
            if args.auto_scale_batch_size:
                print("Automatically found batch size")
                print("Replace --auto_scale_batch_size with")
                print(f"--batch_size {model.batch_size}")
                break
        if args.model not in NO_TRAIN_MODELS:
            trainer.fit(model=model, train_dataloaders=datamodule_i)
        if args.fast_dev_run or args.model in NO_TRAIN_MODELS:
            test_metric = trainer.test(model, dataloaders=datamodule)
        elif not args.no_early_stopping:
            test_metric = trainer.test(
                ckpt_path=model_checkpoint.best_model_path,
                dataloaders=datamodule,
                dataloaders=datamodule,
            )
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
    logs = generate_logs(test_metrics, time_elapsed, args)
    log_to_json(logs)
    logs = generate_logs(test_metrics, time_elapsed, args)
    log_to_json(logs)
    aggregate_logs()
    log_to_gsheet(logs)

    extract_lightning_logs(args)  # Pulls out the one checkpoint we want


if __name__ == "__main__":
    main()
