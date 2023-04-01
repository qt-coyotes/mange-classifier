import csv
import glob
import json
from datetime import datetime, timedelta
import argparse


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


def aggregate_logs():
    paths = glob.glob("logs_*.json")
    rows = [
        [],  # Model
        [],  # Dataset
        [],  # Evaluation Strategy
        [],  # Loss Function
        [],  # Internal K
        [],  # Parameters (M)
        [],  # Data Augmentation
        [],  # Pretrained
        [],  # Early Stopping Patience
        [],  # Epochs
        [],  # Batch Size
        [],  # Learning Rate
        [],
        [],  # Expected Cost (50 : 1)
        [],  # Expected Cost (10 : 1)
        [],  # Expected Cost (5 : 1)
        [],  # F2 Score
        [],  # F1 Score 🤮
        [],  # Recall
        [],  # Precision
        [],  # Average Precision
        [],  # Accuracy 🤮
        [],  # AUROC 🤮
        [],  # Confusion Matrix
        [],  # Loss
        [],
        [],  # Training Time (h:mm:ss)
    ]
    for path in paths:
        with open(path) as f:
            logs = json.load(f)
        model = logs["args"]["model"]
        if model == "ResNet":
            model = logs["args"]["resnet_model"]
        rows[0].append(model)
        rows[1].append("qt coyotes merged dataset v11")
        rows[2].append("StratifiedGroupKFold")
        criterion = logs["args"]["criterion"]
        if criterion == "ExpectedCostLoss":
            criterion += str(logs["args"]["criterion_cfn"])
        elif criterion == "wBCELoss":
            criterion += str(logs["args"]["criterion_pos_weight"])
        rows[3].append(criterion)
        rows[4].append(logs["args"]["internal_k"])
        rows[5].append("?")
        rows[6].append(not logs["args"]["no_data_augmentation"])
        rows[7].append(not logs["args"]["nonpretrained"])
        rows[8].append(logs["args"]["patience"])
        rows[9].append(logs["args"]["max_epochs"])
        rows[10].append(logs["args"]["batch_size"])
        rows[11].append(logs["args"]["learning_rate"])
        rows[12].append("")
        rows[13].append(logs["cv_metrics"]["ExpectedCost50"])
        rows[14].append(logs["cv_metrics"]["ExpectedCost10"])
        rows[15].append(logs["cv_metrics"]["ExpectedCost5"])
        rows[16].append(logs["cv_metrics"]["F2"])
        rows[17].append(logs["cv_metrics"]["F1"])
        rows[18].append(logs["cv_metrics"]["Recall"])
        rows[19].append(logs["cv_metrics"]["Precision"])
        rows[20].append(logs["cv_metrics"]["AveragePrecision"])
        rows[21].append(logs["cv_metrics"]["Accuracy"])
        rows[22].append(logs["cv_metrics"]["AUROC"])
        rows[23].append(
            json.dumps(logs["cv_metrics"]["metric_confusion_matrix"])
        )
        rows[24].append(logs["cv_metrics"]["loss"])
        rows[25].append("")
        rows[26].append(logs["time_elapsed"])

    with open("logs.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
