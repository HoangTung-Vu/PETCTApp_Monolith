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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --frozen --no-dev

COPY weights/ /app/weights/

COPY src/ src/
COPY main.py .

EXPOSE 8103

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8103"]
