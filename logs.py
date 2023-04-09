import argparse
import csv
import glob
import json
import os
import random
import shutil
import uuid
from datetime import datetime, timedelta
from functools import lru_cache

import git
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

    logs = {
        "args": vars(args),
        "cv_metrics": cv_metrics,
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

    row.append(logs["cv_metrics"]["ExpectedCost5"])
    row.append(logs["cv_metrics"]["Precision"])
    row.append(logs["cv_metrics"]["Recall"])
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

    row.append(logs["cv_metrics"]["ExpectedCost50"])
    row.append(logs["cv_metrics"]["ExpectedCost10"])
    row.append(logs["cv_metrics"]["F2"])
    row.append(logs["cv_metrics"]["F1"])
    row.append(logs["cv_metrics"]["AveragePrecision"])
    row.append(logs["cv_metrics"]["AUROC"])
    row.append(logs["cv_metrics"]["Accuracy"])
    row.append(str(logs["cv_metrics"]["loss"]))
    row.append(None)

    row.append(logs["args"]["metadata_path"])

    with open(logs["args"]["metadata_path"]) as f:
        coco = json.load(f)

    row.append(coco["info"]["version"])

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


def log_to_gsheet(logs, row):
    creds = get_gsheet_creds()

    if (
        logs["args"]["metadata_path"]
        == "data/CHIL/CHIL_uwin_mange_Marit_07242020.json"
    ):
        RANGE_NAME = "CHIL!A1:A1"
    else:
        RANGE_NAME = "v17!A1:A1"

    try:
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()
        sheet.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME,
            body={
                "majorDimension": "ROWS",
                "values": [row],
            },
            valueInputOption="USER_ENTERED",
        ).execute()

    except HttpError as err:
        print(err)


def extract_lightning_logs(args):
    if args.no_save_checkpoint:
        print("Not saving this checkpoint, as specified in args")
        return

    if not os.path.exists(STRIP_DIR):
        os.mkdir(STRIP_DIR)

    with open(f"{STRIP_DIR}/checkpoints.tsv", "w") as f:
        f.write("time\tmodel\tcheckpoint\tmessage\n")

    logs = list(
        filter(lambda x: x.startswith("version_"), os.listdir("lightning_logs"))
    )

    if len(logs) == 0:
        return

    retain = random.choice(logs)  # Hope it's a directory I guess?
    log_dir_name = uuid.uuid1()
    t = str(datetime.today()).split(".")[0].replace(" ", "_").replace(":", "-")

    shutil.move(f"lightning_logs/{retain}", f"{STRIP_DIR}/{log_dir_name}")

    with open(f"{STRIP_DIR}/checkpoints.tsv", "a") as f:
        f.write(f"{t}\t{str(args.model)}\t{log_dir_name}\t{args.message}\n")
