# syntax=docker/dockerfile:1.7-labs
ARG build_image
ARG base_image

FROM $build_image AS build-image

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG hayhooks_version

# Shallow clone Hayhooks repo, we'll install from the local sources
RUN git clone --depth=1 --single-branch --branch="${hayhooks_version}" https://github.com/deepset-ai/hayhooks.git /opt/hayhooks
WORKDIR /opt/hayhooks

# Use a virtualenv we can copy over the next build stage
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install .


FROM $base_image AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

COPY --from=build-image /opt/venv /opt/venv

ARG pipelines_dir
RUN mkdir -p "$pipelines_dir"
ENV HAYHOOKS_PIPELINES_DIR=$pipelines_dir

ARG additional_python_path
RUN mkdir -p "$additional_python_path"
ENV HAYHOOKS_ADDITIONAL_PYTHON_PATH=$additional_python_path

EXPOSE 1416

ENV PATH="/opt/venv/bin:$PATH"

CMD ["hayhooks", "run", "--host", "0.0.0.0"]
