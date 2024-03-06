ARG build_image
ARG base_image

FROM $build_image AS build-image

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git

ARG hayhooks_version

# Shallow clone Hayhooks repo, we'll install from the local sources
RUN git clone --depth=1 --branch=${hayhooks_version} https://github.com/deepset-ai/hayhooks.git /opt/hayhooks
WORKDIR /opt/hayhooks

# Use a virtualenv we can copy over the next build stage
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .


FROM $base_image AS final

COPY --from=build-image /opt/venv /opt/venv

ARG pipelines_dir
RUN mkdir -p $pipelines_dir
ENV HAYHOOKS_PIPELINES_DIR=$pipelines_dir

ARG additional_python_path
RUN mkdir -p $additional_python_path
ENV HAYHOOKS_ADDITIONAL_PYTHONPATH=$additional_python_path

EXPOSE 1416

ENV PATH="/opt/venv/bin:$PATH"

CMD ["hayhooks", "run", "--host", "0.0.0.0"]
