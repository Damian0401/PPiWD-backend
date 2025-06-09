FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies using apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    libatlas-base-dev \
    gfortran \
    git \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app

# Cache pip installs
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python3", "main.py" ]
