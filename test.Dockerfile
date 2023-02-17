FROM squarefactory/archipel-base-cpu:latest

RUN pip install \
  albumentations==1.3.0 \
  matplotlib==3.5.1 \
  numpy==1.23.0 \
  torch==1.13.0 \
  torchvision==0.14.0

ARG FACE_PIXELIZER=/opt/face_pixelizer
RUN mkdir ${FACE_PIXELIZER}
COPY face_pixelizer.py ${FACE_PIXELIZER}
COPY retinaface.py ${FACE_PIXELIZER}
COPY utils.py ${FACE_PIXELIZER}
ENV PYTHONPATH="${FACE_PIXELIZER}:${PYTHONPATH}"
COPY "retinaface_mobilenet_0.25.pth" "${FACE_PIXELIZER}/retinaface_mobilenet_0.25.pth"
