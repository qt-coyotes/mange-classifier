FROM nvcr.io/nvidia/pytorch:23.02-py3

COPY . /mange-classifier

WORKDIR /mange-classifier

RUN pip install -r requirements.txt

RUN dvc remote modify gdrive gdrive_use_service_account true

RUN dvc remote modify gdrive --local gdrive_service_account_json_file_path service-account-key.json

RUN train.py
