FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.13 via deadsnakes PPA + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv python3.13-dev \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# No uv needed, directly using hardcoded .venv

WORKDIR /app

# Copy pre-configured virtual environment
COPY .venv/ .venv/

RUN rm -f .venv/bin/python .venv/bin/python3* && \
    ln -s /usr/bin/python3.13 .venv/bin/python && \
    ln -s /usr/bin/python3.13 .venv/bin/python3

ENV PYTHONPATH="/app/.venv/lib/python3.13/site-packages:${PYTHONPATH}"

COPY weights/ /app/weights/

COPY src/ src/
COPY main.py .


EXPOSE 8104

CMD ["./.venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8104"]
