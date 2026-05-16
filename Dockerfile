# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

FROM ${BASE_IMAGE} AS base
ARG TORCH_INDEX_URL
ARG DEBIAN_FRONTEND=noninteractive
ENV UV_EXTRA_INDEX_URL=${TORCH_INDEX_URL}
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential clang ccache \
    libomp-dev libglfw3 libgl1-mesa-dev python3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --no-modify-path \
    && curl -LsSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /work
CMD ["bash"]

FROM base AS jupyter
RUN uv pip install --system --break-system-packages jupyterlab
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
