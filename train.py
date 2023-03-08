import argparse
from model import Module
from data import StratifiedGroupKFoldDataModule
from lightning.pytorch import Trainer
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        type=int,
        default=5
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/CHIL/images"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/CHIL/CHIL_uwin_mange_Marit_07242020.json"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="mps"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10
    )
    args = parser.parse_args()

    metrics = []
    for i in range(args.k):
        model = Module()
        trainer = Trainer(
            accelerator=args.accelerator,
            devices=1,
            limit_train_batches=100,
            max_epochs=args.max_epochs,
            log_every_n_steps=1
        )
        datamodule = StratifiedGroupKFoldDataModule(
            args.k,
            i,
            Path(args.data_path),
            Path(args.metadata_path),
            args.batch_size,
            args.num_workers
        )
        trainer.fit(
            model=model,
            train_dataloaders=datamodule
        )
        metric = trainer.test(
            model=model,
            dataloaders=datamodule
        )
        metrics.append(metric)
        print(metric)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
