FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

SHELL ["/bin/bash", "-lc"]

# 設置工作目錄
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        redis-server \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Run it in container
# `bash setup_lm.sh`

RUN chmod +x setup.sh && bash setup.sh