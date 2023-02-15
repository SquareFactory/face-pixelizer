FROM alpineintuition/archipel-base-cpu:latest

COPY deploy/deploy_requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm requirements.txt

ARG FACE_PIXELIZER=/opt/face_pixelizer
RUN mkdir ${FACE_PIXELIZER} && \
    mkdir ${FACE_PIXELIZER}/retina && \
    mkdir ${FACE_PIXELIZER}/deploy

COPY deploy/face_pixelizer.py ${FACE_PIXELIZER}/deploy/
COPY retina ${FACE_PIXELIZER}/retina

ENV PYTHONPATH="${FACE_PIXELIZER}:${PYTHONPATH}"
COPY weights/"retinaface_mobilenet_0.25.pth" "${FACE_PIXELIZER}/retinaface_mobilenet_0.25.pth"
