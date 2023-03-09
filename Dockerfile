FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN mkdir -p /mange-classifier
WORKDIR /mange-classifier
COPY ./requirements.txt /mange-classifier
RUN pip install -U -r requirements.txt

CMD make
