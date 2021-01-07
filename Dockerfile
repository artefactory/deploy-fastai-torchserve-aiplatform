FROM pytorch/torchserve:0.2.0-cpu

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-setuptools \
    python3-pip

COPY requirements.txt /home/model-server/

RUN pip install --upgrade pip
RUN pip install -r /home/model-server/requirements.txt

COPY torch_serve/handler.py torch_serve/config.py /home/model-server/

COPY model/fastai_cls_weights.pth model/spm/spm.model model/spm/spm.vocab /home/model-server/

RUN torch-model-archiver \
  --model-name=fastai_model \
  --version=1.0 \
  --serialized-file=/home/model-server/fastai_cls_weights.pth \
  --extra-files=/home/model-server/spm.model,/home/model-server/spm.vocab,/home/model-server/config.py \
  --handler=/home/model-server/handler.py \
  --export-path=/home/model-server/model-store/


CMD ["torchserve", \
     "--start", \
     "--ncs", \
     "--model-store", \
     "/home/model-server/model-store", \
     "--models", \
     "fastai_model.mar"]
