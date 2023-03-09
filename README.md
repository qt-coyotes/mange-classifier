# mange-classifier

## Installation

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
dvc pull
```

## Training

```bash
make run
```

Observe outputted `cv-metrics-*.json` for CV metrics.

## DVC

A DVC remote is setup with Google Drive, to pull data use `dvc pull`.
