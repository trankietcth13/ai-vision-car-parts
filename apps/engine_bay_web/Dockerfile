# CPU image. For an NVIDIA GPU server, see README.md (base image with CUDA + ENGINE_BAY_DEVICE=0).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ENGINE_BAY_HOST=0.0.0.0 ENGINE_BAY_PORT=7860 ENGINE_BAY_DEVICE=cpu \
    YOLO_CONFIG_DIR=/tmp/ultralytics GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY models ./models
COPY config ./config
COPY examples ./examples

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://127.0.0.1:7860/healthz || exit 1
CMD ["python", "app.py"]
