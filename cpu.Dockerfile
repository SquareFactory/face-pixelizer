FROM alpineintuition/archipel-base-cpu:latest

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ARG FACE_PIXELIZER=/opt/face_pixelizer
RUN mkdir ${FACE_PIXELIZER}
COPY face_pixelizer.py ${FACE_PIXELIZER}
COPY retinaface.py ${FACE_PIXELIZER}
COPY utils.py ${FACE_PIXELIZER}
ENV PYTHONPATH="${FACE_PIXELIZER}:${PYTHONPATH}"
COPY "retinaface_mobilenet_0.25.pth" "${FACE_PIXELIZER}/retinaface_mobilenet_0.25.pth"
