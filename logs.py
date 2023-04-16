import argparse
import csv
import glob
import json
import os
from datetime import datetime, timedelta
from functools import lru_cache
import scipy.stats

import git
import numpy as np
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1nFRtoKX3q4MXsyvjImYz-_jdtw4LmSAaZPXffv1C2js"
STRIP_DIR = "stripped_logs"


def generate_logs(
    test_metrics, time_elapsed: timedelta, args: argparse.Namespace
):
    cv_metrics = {"metric_confusion_matrix": []}
    for test_metric in test_metrics:
        test_metric = test_metric[0]
        test_metric_metric = test_metric["test_metric"]
        for key, value in test_metric_metric.items():
            key = key.replace("Binary", "")
            if key not in cv_metrics:
                cv_metrics[key] = []
            cv_metrics[key].append(value.item())
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
        if "loss" not in cv_metrics:
            cv_metrics["loss"] = []
        cv_metrics["loss"].append(test_metric["test_loss"])

    confidence = 0.95
    cv_stats = {}
    for metric in cv_metrics:
        if metric == "metric_confusion_matrix":
            continue
        cv_stats[f"{metric}_std"] = np.array(cv_metrics[metric]).std()
        cv_stats[f"{metric}_mean"] = np.array(cv_metrics[metric]).mean()
        cv_stats[f"{metric}_{int(confidence * 100)}_CI"] = scipy.stats.t.interval(
            confidence, len(cv_metrics[metric])-1,
            loc=cv_stats[f"{metric}_mean"],
            scale=scipy.stats.sem(cv_metrics[metric])
        )

    logs = {
        "args": vars(args),
        "cv_metrics": cv_metrics,
        "cv_stats": cv_stats,
        "time_elapsed": str(time_elapsed),
        "timestamp": datetime.now().isoformat(),
    }
    print(logs)
    if args.fast_dev_run:
        return
    return logs


def log_to_json(logs):
    with open(f"logs_{logs['timestamp'].replace(':', '-')}.json", "w") as f:
        json.dump(logs, f, indent=4)


def aggregate_logs():
    paths = glob.glob("logs_*.json")
    paths.sort()
    rows = []
    for path in paths:
        with open(path) as f:
            logs = json.load(f)
        row = get_row(logs)
        rows.append(row)

    with open("logs.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def get_row(logs):
    row = []

    repo = git.Repo()
    ref = repo.head.ref
    message = logs["args"].get("message")
    if not message:
        message = ref.commit.message
    message = message.strip()
    row.append(message)
    timestamp = logs.get("timestamp")
    if not timestamp:
        timestamp = datetime.now().isoformat()

    row.append(timestamp)
    row.append(
        f"{os.environ.get('GITHUB_SERVER_URL')}/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
    )
    row.append(None)

    row.append(logs["cv_stats"]["ExpectedCost5_mean"])
    row.append(logs["cv_stats"]["Precision_mean"])
    row.append(logs["cv_stats"]["Recall_mean"])
    row.append(json.dumps(logs["cv_metrics"]["metric_confusion_matrix"]))
    row.append(logs["time_elapsed"])
    row.append(None)

    model = logs["args"]["model"]
    if model == "ResNet":
        model = logs["args"]["resnet_model"]
    elif model == "DenseNet":
        model = logs["args"]["densenet_model"]
    elif model == "ViT":
        model = logs["args"]["vit_model"]
    elif model == "YOLO":
        model = logs["args"]["yolo_model"]
    row.append(model)

    criterion = logs["args"]["criterion"]
    if criterion == "ExpectedCostLoss":
        criterion += str(logs["args"]["criterion_cfn"])
    elif criterion == "wBCELoss":
        criterion += str(logs["args"]["criterion_pos_weight"])
    row.append(criterion)
    row.append(logs["args"]["internal_k"])
    row.append(None)
    row.append(not logs["args"]["no_data_augmentation"])
    row.append(not logs["args"]["nonpretrained"])
    row.append(logs["args"]["patience"])
    row.append(logs["args"]["max_epochs"])
    row.append(logs["args"]["batch_size"])
    row.append(logs["args"]["learning_rate"])
    row.append(ref.commit.hexsha)
    row.append(None)

    row.append(logs["cv_stats"]["ExpectedCost50_mean"])
    row.append(logs["cv_stats"]["ExpectedCost10_mean"])
    row.append(logs["cv_stats"]["F2_mean"])
    row.append(logs["cv_stats"]["F1_mean"])
    row.append(logs["cv_stats"]["AveragePrecision_mean"])
    row.append(logs["cv_stats"]["AUROC_mean"])
    row.append(logs["cv_stats"]["Accuracy_mean"])
    row.append(str(logs["cv_stats"]["loss_mean"]))
    row.append(None)

    row.append(logs["args"]["metadata_path"])

    with open(logs["args"]["metadata_path"]) as f:
        coco = json.load(f)

    row.append(coco["info"]["version"])
    row.append(None)

    row.extend(logs["cv_stats"]["ExpectedCost5_95_CI"])
    row.extend(logs["cv_stats"]["Precision_95_CI"])
    row.extend(logs["cv_stats"]["Recall_95_CI"])
    row.extend(logs["cv_stats"]["ExpectedCost50_95_CI"])
    row.extend(logs["cv_stats"]["ExpectedCost10_95_CI"])
    row.extend(logs["cv_stats"]["F2_95_CI"])
    row.extend(logs["cv_stats"]["F1_95_CI"])
    row.extend(logs["cv_stats"]["AveragePrecision_95_CI"])
    row.extend(logs["cv_stats"]["AUROC_95_CI"])
    row.extend(logs["cv_stats"]["Accuracy_95_CI"])
    row.append(None)

    row.extend(logs["cv_metrics"]["ExpectedCost5"])

    return row


@lru_cache(maxsize=1)
def get_gsheet_creds():
    if os.environ.get("GITHUB_ACTIONS"):
        with open("service-account-key.json", "w") as f:
            f.write(os.environ.get("GDRIVE_CREDENTIALS_DATA"))
    creds = Credentials.from_service_account_file(
        "service-account-key.json",
        scopes=SCOPES,
    )
    return creds


def log_to_gsheet(row, gsheet_range):
    creds = get_gsheet_creds()

    try:
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()
        sheet.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=gsheet_range,
            body={
                "majorDimension": "ROWS",
                "values": [row],
            },
            valueInputOption="USER_ENTERED",
        ).execute()

    except HttpError as err:
        print(err)
