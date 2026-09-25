# syntax=docker/dockerfile:1.7
#
# Targets:
#   runtime  GPU image (VARIANT=gpu, default): core + learned deps
#   dev      CPU image (VARIANT=cpu): core + dev tools, for tests and linting
#
# No CUDA toolkit is needed in the image: torch cu128 wheels bundle CUDA/cuDNN
# and pycolmap-cuda12 pulls the CUDA runtime from pip. The nvidia/cuda base only
# carries the env vars the NVIDIA container runtime reads.

ARG VARIANT=gpu

FROM nvidia/cuda:12.8.1-base-ubuntu24.04 AS base-gpu
FROM ubuntu:24.04 AS base-cpu

FROM base-${VARIANT} AS base

# Match the host user so files written to bind mounts are owned by them.
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    XDG_CACHE_HOME=/cache \
    TORCH_HOME=/cache/torch

# python3.12 is Ubuntu 24.04's default. git: hloc/LightGlue are git deps.
# libgl1/libglib2.0-0/libgomp1/libegl1: runtime libs for OpenCV and Open3D;
# libsm6: pycolmap._core links libSM/libICE.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv git ca-certificates \
        libgl1 libglib2.0-0 libgomp1 libegl1 libsm6

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

# ubuntu:24.04 ships a default `ubuntu` user at 1000:1000; drop it so UID/GID
# can be reused.
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupadd -g "${GID}" app \
    && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash app \
    && mkdir -p /app /opt/venv /cache \
    && chown -R app:app /app /opt/venv /cache

USER app
WORKDIR /app

# ---------------------------------------------------------------------------
FROM base AS runtime

# No uv cache here: a cache mount keeps every downloaded wheel (~9 GB with the
# CUDA wheels) in the builder cache on the shared GPU server. The dependency
# layer below is still reused as long as pyproject.toml/uv.lock are unchanged;
# only dependency changes re-download.
ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

# Dependencies first (cached layer), project last.
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev --group learned

COPY --chown=app:app . /app
RUN uv sync --frozen --no-dev --group learned

CMD ["bash"]

# ---------------------------------------------------------------------------
FROM base AS dev

ARG UID=1000
ARG GID=1000

RUN --mount=type=cache,target=/cache/uv,uid=${UID},gid=${GID} \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project

COPY --chown=app:app . /app
RUN --mount=type=cache,target=/cache/uv,uid=${UID},gid=${GID} \
    uv sync --frozen

CMD ["pytest"]
