FROM nvcr.io/nvidia/pytorch:22.02-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing && \
    apt-get install -y \
      curl \
      libsm6 \
      libxext6 \
      python3-virtualenv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



COPY train/train_requirements.txt requirements.txt
RUN pip install retinaface_pytorch==0.0.8 --no-deps && \
    pip install opencv-python wget==3.2 && \
    pip install -r requirements.txt && \
    rm requirements.txt