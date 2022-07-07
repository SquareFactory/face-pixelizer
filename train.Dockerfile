FROM alpineintuition/archipel-base-gpu:latest

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN mkdir opt/face_pixelizer
COPY /src /opt/face_pixelizer/src
COPY train_config_gpu.yaml opt/train_config_gpu.yaml
ENV PYTHONPATH="opt/face_pixelizer:${PYTHONPATH}"
