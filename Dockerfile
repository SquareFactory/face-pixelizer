FROM alpineintuition/archipel-base-gpu:latest

RUN pip install \
      albumentations==0.5.1 \
      numpy==1.19.2 \
      torch==1.7.0 \
      torchvision==0.8.1

ARG FACE_PIXELIZER=/opt/face_pixelizer
RUN mkdir ${FACE_PIXELIZER}
COPY retinaface.py ${FACE_PIXELIZER}
COPY utils.py ${FACE_PIXELIZER}
ENV PYTHONPATH="${FACE_PIXELIZER}:${PYTHONPATH}"

RUN wget -O "${FACE_PIXELIZER}/retinaface_mobilenet_0.25.pth" \
      --user alpine --password kdMHG6VsyDgwcvg3RdaUQiDhoqdmn4 \
      http://alpineintuition.com/i2-storage/private/retinaface_mobilenet_0.25.pth
