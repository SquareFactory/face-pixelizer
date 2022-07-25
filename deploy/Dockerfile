FROM alpineintuition/archipel-base-cpu:latest
ARG FACE_PIXELIZER=/opt/face_pixelizer
RUN mkdir ${FACE_PIXELIZER}
COPY deploy ${FACE_PIXELIZER}
RUN pip install -r ${FACE_PIXELIZER}/requirements.txt
ENV PYTHONPATH="${FACE_PIXELIZER}:${PYTHONPATH}"
