# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
ARG NCCL_PACKAGES="libnccl2 libnccl-dev"
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

FROM ${BASE_IMAGE}
ARG NCCL_PACKAGES
ARG TORCH_INDEX_URL
ARG DEBIAN_FRONTEND=noninteractive

ENV UV_EXTRA_INDEX_URL=${TORCH_INDEX_URL}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential clang ccache \
        libomp-dev libglfw3 libgl1-mesa-dev \
        python3.13 python3.13-dev \
        ${NCCL_PACKAGES} \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --no-modify-path \
 && curl -LsSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin \
 && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /work
CMD ["bash"]
