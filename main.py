import argparse
import gc
import itertools
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
    get_row,
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
from models.densenet import DenseNetModel
from models.random import RandomModel
from models.resnet import ResNetModel
from models.vit import ViTModel
from models.yolo import YoloModel
from pytorch_lightning import LightningDataModule


NO_TRAIN_MODELS = {
    "AllPositive": (AllPositiveModel, None),
    "AllNegative": (AllNegativeModel, None),
    "Random": (RandomModel, None),
    "SuperLearner": (None, None),
}

SUPER_LEARNER_MODELS = {
    "ResNet18": (ResNetModel, 18),
    "ResNet34": (ResNetModel, 34),
    "ResNet50": (ResNetModel, 50),
    # "ResNet101": (ResNetModel, 101),
    # "ResNet152": (ResNetModel, 152),
    "ViT-B/16": (ViTModel, "B/16"),
    # "ViT-B/32": (ViTModel, "B/32"),
    # "ViT-L/16": (ViTModel, "L/16"),
    # "ViT-L/32": (ViTModel, "L/32"),
    "DenseNet121": (DenseNetModel, 121),
    # "DenseNet161": (DenseNetModel, 161),
    # "DenseNet169": (DenseNetModel, 169),
    # "DenseNet201": (DenseNetModel, 201),
    # "yolov8n-cls.pt": (YoloModel, "yolov8n-cls.pt"),
    "yolov8s-cls.pt": (YoloModel, "yolov8s-cls.pt"),
    # "yolov8m-cls.pt": (YoloModel, "yolov8m-cls.pt"),
    # "yolov8l-cls.pt": (YoloModel, "yolov8l-cls.pt"),
    # "yolov8x-cls.pt": (YoloModel, "yolov8x-cls.pt"),
}

MODELS = {**NO_TRAIN_MODELS, **SUPER_LEARNER_MODELS}

LEARNING_RATES = {
    # 0.001,
    0.0001,
    # 0.00001,
    "--auto_lr_find",
}

BATCH_SIZES = {
    # 16,
    32,
}

CRITERIONS = {
    "awBCELoss",
    "BCELoss",
    "ExpectedCostLoss",
    # "wBCELoss",
    # "MacroSoftFBetaLoss",
    # "SurrogateFBetaLoss",
    # "HybridLoss",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    group = parser.add_argument_group("qt.coyote")
    group.add_argument(
        "--model",
        help="Which model to use",
        type=str,
        choices=list(MODELS.keys()),
        default="ResNet",
    )
    group.add_argument(
        "--criterion",
        help="Which criterion to use",
        type=str,
        choices=CRITERIONS,
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
    args = parser.parse_args(argv)
    if args.accelerator is None:
        args.accelerator = "auto"

    return args


def main():
    args = parse_args()
    print(args)
    external_cross_validation(args)
    # train_final_model(args)


def model_from_args(args: argparse.Namespace, datamodule_i: LightningDataModule):
    torch.backends.cudnn.deterministic = not args.nondeterministic
    Model, architecture = MODELS[args.model]
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
    if architecture is not None:
        model = Model(criterion, args, architecture=architecture)
    else:
        model = Model(criterion, args)
    if args.compile and isinstance(trainer.accelerator, CUDAAccelerator):
        model = torch.compile(model)
    if args.auto_scale_batch_size or args.auto_lr_find:
        datamodule_i.setup(None)
        X, _ = next(iter(datamodule_i.train_dataloader()))
        _ = model(X)
        trainer.tune(model, datamodule=datamodule_i)
        print(
            f"Automatically found learning rate: {model.learning_rate}"
        )
        args.learning_rate = model.learning_rate
    if args.auto_scale_batch_size:
        print(f"Automatically found batch size: P={model.batch_size}")
        args.batch_size = model.batch_size
    if not args.max_epochs:
        args.max_epochs = 100
    return model, trainer, model_checkpoint


def internal_cross_validation(datamodule: LightningDataModule):
    best_EC5 = torch.inf
    best_args = None
    best_checkpoint = None
    argvs = list(itertools.product(
        ("--model",), SUPER_LEARNER_MODELS,
        ("--criterion",), CRITERIONS,
        ("--learning_rate",), map(str, LEARNING_RATES),
        ("--batch_size",), map(str, BATCH_SIZES),
        ("--no_crop", ""),
        ("--no_data_augmentation", ""),
        ("--no_tabular_features", ""),
    ))
    print(f"Number of hyperparameter configurations: {len(argvs)}")
    for c, argv in enumerate(argvs):
        argv = list(argv)
        if "--auto_lr_find" in argv:
            argv.remove("--learning_rate")
        argv = list(filter(len, argv))
        print(f"Hyperparameter configuration: {c}/{len(argvs)}")
        args = parse_args(argv)
        model = model_from_args(args, datamodule)
        model, trainer, model_checkpoint = model_from_args(args, datamodule)
        # internal leakage of pos_weight and early stopping
        trainer.fit(model, datamodule=datamodule)
        EC5 = model_checkpoint.best_model_score
        print(f"EC5: {EC5}")
        log_to_gsheet([f"{EC5}", f"{c / len(argvs)}", f"{c}", f"{len(argvs)}", f"{' '.join(argv)}"], "SuperLearner!A1:A1")
        if EC5 < best_EC5:
            best_EC5 = EC5
            best_args = args
            best_checkpoint = model_checkpoint
    print(f"Best EC5: {best_EC5}")
    print(f"Best args: {best_args}")
    log_to_gsheet([f"{best_EC5}", f"{best_args}"], "SuperLearner!A1:A1")
    return best_args, best_checkpoint


def external_cross_validation(args: argparse.Namespace):
    start_time = time.perf_counter()
    test_metrics = []
    datamodule = StratifiedGroupKFoldDataModule(args)
    args_copy = args
    for datamodule_i in datamodule:
        seed_everything(args.random_state, workers=True)
        if args_copy.model == "SuperLearner":
            args, model_checkpoint = internal_cross_validation(datamodule_i)
            model, trainer, _ = model_from_args(args, datamodule_i)
        else:
            model, trainer, model_checkpoint = model_from_args(args, datamodule_i)
            if args.model not in NO_TRAIN_MODELS:
                trainer.fit(model=model, train_dataloaders=datamodule_i)
            if args.fast_dev_run or args.model in NO_TRAIN_MODELS:
                test_metric = trainer.test(model, dataloaders=datamodule)
        if not args.no_early_stopping:
            test_metric = trainer.test(
                model=model,
                ckpt_path=model_checkpoint.best_model_path,
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
    logs = generate_logs(test_metrics, time_elapsed, args_copy)
    log_to_json(logs)
    aggregate_logs()
    row = get_row(logs)
    if (
        logs["args"]["metadata_path"]
        == "data/CHIL/CHIL_uwin_mange_Marit_07242020.json"
    ):
        gsheet_range = "CHIL!A1:A1"
    else:
        gsheet_range = "v17!A1:A1"
    log_to_gsheet(row, gsheet_range)

    extract_lightning_logs(args_copy)  # Pulls out the one checkpoint we want


def train_final_model(args: argparse.Namespace):
    pass


if __name__ == "__main__":
    main()
