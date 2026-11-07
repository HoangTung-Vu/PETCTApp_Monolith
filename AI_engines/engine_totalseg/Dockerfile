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

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8105"]
