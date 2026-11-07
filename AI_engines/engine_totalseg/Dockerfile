# TotalSegmentator engine — FastAPI server for TotalSegmentator task "total" at default
# settings (5 part models, 1.5 mm, 117 classes). Weights are baked in: no internet needed
# at runtime.
#
# Base: official PyTorch 2.7.1 + CUDA 12.6 + cuDNN 9. Its kernels cover sm_50..sm_90,
# including Turing (Quadro RTX 8000, sm_75) and Ampere (RTX 3090, sm_86). Host needs an
# NVIDIA driver >= 525 (CUDA 12 minor-version compatibility). Runs on CPU without --gpus.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    TOTALSEG_HOME_DIR=/app/totalseg

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Weights layer comes before the code so code edits don't re-download them.
COPY prepare_weights.py .
RUN python prepare_weights.py

COPY src/ src/
COPY main.py .

EXPOSE 8105

# The server process (main.py) only runs FastAPI. Each job runs TotalSegmentator in a
# separate worker process (python -m src.worker, from /app), started and watched by
# src/runner.py. ENGINE_STALL_TIMEOUT / ENGINE_MAX_ATTEMPTS tune the watchdog.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8105"]

# Code-only update (requirements and weights unchanged): add a thin layer on top of the
# pushed image instead of rebuilding 14 GB, from the repo root:
#   printf 'FROM hoangtungmvu/totalseg-engine:2.18.0\nCOPY src/ src/\nCOPY main.py .\n' \
#     | docker build -t hoangtungmvu/totalseg-engine:2.18.0 -f - AI_engines/engine_totalseg
#   docker tag hoangtungmvu/totalseg-engine:2.18.0 hoangtungmvu/totalseg-engine:latest
#   docker push hoangtungmvu/totalseg-engine:2.18.0 && docker push hoangtungmvu/totalseg-engine:latest
# A file deleted from src/ stays in the image that way: rebuild from this Dockerfile then.
