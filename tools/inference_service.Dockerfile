# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/huggingface \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/anonymizer/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH=/opt/anonymizer/.venv/bin:$PATH

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/anonymizer
COPY . .

RUN uv python install 3.12 \
    && uv sync --frozen --python 3.12 --no-default-groups --group local-models

CMD ["sleep", "infinity"]
