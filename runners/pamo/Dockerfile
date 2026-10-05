# Upstream image contains PaMO's three compiled CUDA stages and Python 3.10.
# Build from the PhysicsOS repository root; execute on an NVIDIA GPU host.
ARG PAMO_BASE_IMAGE=sarahwei0210/pamo:0.0.2
FROM ${PAMO_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libopengl0 libegl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY runners/pamo/requirements.txt /opt/physicsos-pamo/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/physicsos-pamo/requirements.txt
COPY physicsos/backends/pamo_worker.py physicsos/backends/surface_mesh.py /opt/physicsos-pamo/
WORKDIR /job
ENTRYPOINT ["python", "/opt/physicsos-pamo/pamo_worker.py"]
CMD ["--help"]
