# Hayhooks

[![PyPI - Version](https://img.shields.io/pypi/v/hayhooks.svg)](https://pypi.org/project/hayhooks)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hayhooks.svg)](https://pypi.org/project/hayhooks)
[![Docker image release](https://github.com/deepset-ai/hayhooks/actions/workflows/docker.yml/badge.svg)](https://github.com/deepset-ai/hayhooks/actions/workflows/docker.yml)
[![Tests](https://github.com/deepset-ai/hayhooks/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/hayhooks/actions/workflows/tests.yml)

**Table of Contents**

- [Install the package](#install-the-package)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [CLI Commands](#cli-commands)
- [Start hayhooks](#start-hayhooks)
- [Deploy a pipeline](#deploy-a-pipeline)
- [OpenAI Compatibility](#openai-compatible-endpoints-generation)
  - [Streaming Responses](#streaming-responses-in-openai-compatible-endpoints)

## Quick start

### Install the package

Start by installing the package:

```console
pip install hayhooks
```

### Configuration

Currently, you can configure **hayhooks** by:

- Set the environment variables in an `.env` file in the root of your project.
- Pass the supported arguments and options to `hayhooks run` command.
- Pass the environment variables to the `hayhooks` command.

#### Environment variables

The following environment variables are supported:

- `HAYHOOKS_HOST`: The host on which the server will listen.
- `HAYHOOKS_PORT`: The port on which the server will listen.
- `HAYHOOKS_PIPELINES_DIR`: The path to the directory containing the pipelines.
- `HAYHOOKS_ROOT_PATH`: The root path of the server.
- `HAYHOOKS_ADDITIONAL_PYTHONPATH`: Additional Python path to be added to the Python path.
- `HAYHOOKS_DISABLE_SSL`: Whether to disable SSL verification when making requests from the CLI.

### CLI commands

The `hayhooks` package provides a CLI to manage the server and the pipelines.
Any command can be run with `hayhooks <command> --help` to get more information.

CLI commands are basically wrappers around the HTTP API of the server. The full API reference is available at [http://HAYSTACK_HOST:HAYSTACK_PORT/docs](http://HAYSTACK_HOST:HAYSTACK_PORT/docs) or [http://HAYSTACK_HOST:HAYSTACK_PORT/redoc](http://HAYSTACK_HOST:HAYSTACK_PORT/redoc).

```shell
hayhooks run     # Start the server
hayhooks status  # Check the status of the server and show deployed pipelines

hayhooks pipeline deploy-files <path_to_files> # Deploy a pipeline using PipelineWrapper
hayhooks pipeline deploy <pipeline_name>       # Deploy a pipeline from a YAML file
hayhooks pipeline undeploy <pipeline_name>     # Undeploy a pipeline
```

### Start hayhooks

To start the server, run:

```console
hayhooks run
```

This will start the hayhooks server on `HAYHOOKS_HOST:HAYHOOKS_PORT`.

### Deploy a pipeline

Now we will deploy the [Chat with website](https://docs.haystack.deepset.ai/docs/pipeline-templates#chat-with-website) pipeline. We have created an example in the [examples/chat_with_website_streaming](examples/chat_with_website_streaming) folder.

In the example folder, we have two files:

- `chat_with_website.yml`: The pipeline definition in YAML format.
- `pipeline_wrapper.py`: A pipeline wrapper that uses the pipeline definition.

The `pipeline_wrapper.py` file must contain an implementation of the `BasePipelineWrapper` class (see [here](src/hayhooks/server/utils/base_pipeline_wrapper.py) for more details).

A minimal wrapper looks like this:

```python
from pathlib import Path
from typing import List
from haystack import Pipeline
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.logger import log


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: List[str], question: str) -> str:
        log.trace(f"Running pipeline with urls: {urls} and question: {question}")
        result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
        return result["llm"]["replies"][0]
```

It contains two methods:

#### setup()

This method will be called when the pipeline is deployed. It should initialize the `self.pipeline` attribute as a Haystack pipeline.

You can initialize the pipeline in many ways:

- Load it from a YAML file.
- Define it inline as a Haystack pipeline code.
- Load it from a [Haystack pipeline template](https://docs.haystack.deepset.ai/docs/pipeline-templates).

#### run_api(Pydantic-compatible arguments) -> (any Pydantic-compatible type)

This method will be used to run the pipeline in API mode, when you call the `{pipeline_name}/run` endpoint.

**You can define the input arguments of the method according to your needs**. The input arguments will be used to generate a Pydantic model that will be used to validate the request body. The same will be done for the response type.

**NOTE**: Since hayhooks will _dynamically_ create the Pydantic models, you need to make sure that the input arguments are JSON-serializable.

To deploy the pipeline, run:

```shell
hayhooks pipeline deploy-files -n chat_with_website examples/chat_with_website
```

This will deploy the pipeline with the name `chat_with_website`. Any error encountered during development will be printed to the console and show in the server logs.

### OpenAI-compatible endpoints generation

`hayhooks` now can automatically generate OpenAI-compatible endpoints if you implement the `run_chat_completion` method in your pipeline wrapper.

This will make hayhooks compatible with fully-featured chat interfaces like [open-webui](https://openwebui.com/).

To enable the automatic generation of OpenAI-compatible endpoints, you need only to implement the `run_chat_completion` method in your pipeline wrapper.

Let's update the previous example to add a streaming response:

```python
from pathlib import Path
from typing import Generator, List, Union
from haystack import Pipeline
from hayhooks.server.pipelines.utils import get_last_user_message
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.logger import log


URLS = ["https://haystack.deepset.ai", "https://www.redis.io", "https://ssi.inc"]


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        ...  # Same as before

    def run_api(self, urls: List[str], question: str) -> str:
        ...  # Same as before

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        log.trace(f"Running pipeline with model: {model}, messages: {messages}, body: {body}")

        question = get_last_user_message(messages)
        log.trace(f"Question: {question}")

        # Plain pipeline run, will return a string
        result = self.pipeline.run({"fetcher": {"urls": URLS}, "prompt": {"query": question}})
        return result["llm"]["replies"][0]
```

Differently from the `run_api` method, the `run_chat_completion` has a **fixed signature** and will be called with the arguments specified in the OpenAI-compatible endpoint.

- `model`: The `name` of the Haystack pipeline which is called.
- `messages`: The list of messages from the chat.
- `body`: The full body of the request.

Some notes:

- Since we have only the user messages as input here, the `question` is extracted from the last user message and the `urls` argument is hardcoded.
- In this example, the `run_chat_completion` method is returning a string, so the `open-webui` will receive a string as response and show the pipeline output in the chat all at once. But we can do better!

### Streaming responses in OpenAI-compatible endpoints

Hayhooks now provides a `streaming_generator` utility function that can be used to stream the pipeline output to the client.

Let's update the previous example to stream the pipeline output:

```python
from pathlib import Path
from typing import Generator, List, Union
from haystack import Pipeline
from hayhooks.server.pipelines.utils import get_last_user_message, streaming_generator
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.logger import log


URLS = ["https://haystack.deepset.ai", "https://www.redis.io", "https://ssi.inc"]


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        ...  # Same as before

    def run_api(self, urls: List[str], question: str) -> str:
        ...  # Same as before

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        log.trace(f"Running pipeline with model: {model}, messages: {messages}, body: {body}")

        question = get_last_user_message(messages)
        log.trace(f"Question: {question}")

        # Streaming pipeline run, will return a generator
        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"fetcher": {"urls": URLS}, "prompt": {"query": question}},
        )
```

Now, if you run the pipeline and call the `{pipeline_name}/chat/completions` endpoint, you will see the pipeline output being streamed to the client and you'll be able to see the output in chunks.
