# Hayhooks

**Hayhooks** makes it easy to deploy and serve [Haystack](https://haystack.deepset.ai/) pipelines as REST APIs.

It provides a simple way to wrap your Haystack pipelines with custom logic and expose them via HTTP endpoints, including OpenAI-compatible chat completion endpoints. With Hayhooks, you can quickly turn your Haystack pipelines into API services with minimal boilerplate code.

[![PyPI - Version](https://img.shields.io/pypi/v/hayhooks.svg)](https://pypi.org/project/hayhooks)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hayhooks.svg)](https://pypi.org/project/hayhooks)
[![Docker image release](https://github.com/deepset-ai/hayhooks/actions/workflows/docker.yml/badge.svg)](https://github.com/deepset-ai/hayhooks/actions/workflows/docker.yml)
[![Tests](https://github.com/deepset-ai/hayhooks/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/hayhooks/actions/workflows/tests.yml)

**Table of Contents**

- [Quick Start with Docker Compose](#quick-start-with-docker-compose-🐳)
- [Quick Start](#quick-start-🚀)
- [Install the package](#install-the-package)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [CORS Settings](#cors-settings)
- [CLI Commands](#cli-commands)
- [Start hayhooks](#start-hayhooks)
- [Deploy a pipeline](#deploy-a-pipeline)
  - [Pipeline Wrapper](#why-a-pipeline-wrapper)
  - [Setup Method](#setup)
  - [Run API Method](#run_api)
  - [Additional Dependencies](#additional-dependencies)
  - [PipelineWrapper development with `overwrite` option](#pipelinewrapper-development-with-overwrite-option)
- [OpenAI Compatibility](#openai-compatible-endpoints-generation)
  - [Using Hayhooks as `open-webui` backend](#using-hayhooks-as-open-webui-backend)
  - [Run Chat Completion Method](#run_chat_completion)
  - [Streaming Responses](#streaming-responses-in-openai-compatible-endpoints)
  - [Integration with haystack OpenAIChatGenerator](#integration-with-haystack-openaichatgenerator)
- [Advanced Usage](#advanced-usage-🔧)
  - [Run Hayhooks Programmatically](#run-hayhooks-programmatically)
- [Deployment Guidelines](#deployment-guidelines-📦)
- [Legacy Features](#legacy-features)
  - [Deploy Pipeline Using YAML](#deploy-a-pipeline-using-only-its-yaml-definition)
- [License](#license)

## Quick start with Docker Compose 🐳

To quickly get started with Hayhooks, we provide a ready-to-use Docker Compose setup with pre-configured integration with [open-webui](https://openwebui.com/).

It's available [here](https://github.com/deepset-ai/hayhooks-open-webui-docker-compose).

## Quick start 🚀

### Install the package

Start by installing the package:

```shell
pip install hayhooks
```

### Configuration

Currently, you can configure Hayhooks by:

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
- `HAYHOOKS_DISABLE_SSL`: Boolean flag to disable SSL verification when making requests from the CLI.
- `HAYHOOKS_SHOW_TRACEBACKS`: Boolean flag to show tracebacks on errors during pipeline execution and deployment.

##### CORS Settings

- `HAYHOOKS_CORS_ALLOW_ORIGINS`: List of allowed origins (default: ["*"])
- `HAYHOOKS_CORS_ALLOW_METHODS`: List of allowed HTTP methods (default: ["*"])
- `HAYHOOKS_CORS_ALLOW_HEADERS`: List of allowed headers (default: ["*"])
- `HAYHOOKS_CORS_ALLOW_CREDENTIALS`: Allow credentials (default: false)
- `HAYHOOKS_CORS_ALLOW_ORIGIN_REGEX`: Regex pattern for allowed origins (default: null)
- `HAYHOOKS_CORS_EXPOSE_HEADERS`: Headers to expose in response (default: [])
- `HAYHOOKS_CORS_MAX_AGE`: Maxium age for CORS preflight responses in seconds (default: 600)

### CLI commands

The `hayhooks` package provides a CLI to manage the server and the pipelines.
Any command can be run with `hayhooks <command> --help` to get more information.

CLI commands are basically wrappers around the HTTP API of the server. The full API reference is available at [//HAYHOOKS_HOST:HAYHOOKS_PORT/docs](http://HAYHOOKS_HOST:HAYHOOKS_PORT/docs) or [//HAYHOOKS_HOST:HAYHOOKS_PORT/redoc](http://HAYHOOKS_HOST:HAYHOOKS_PORT/redoc).

```shell
hayhooks run     # Start the server
hayhooks status  # Check the status of the server and show deployed pipelines

hayhooks pipeline deploy-files <path_to_dir>   # Deploy a pipeline using PipelineWrapper
hayhooks pipeline deploy <pipeline_name>       # Deploy a pipeline from a YAML file
hayhooks pipeline undeploy <pipeline_name>     # Undeploy a pipeline
```

### Start Hayhooks

Let's start Hayhooks:

```shell
hayhooks run
```

This will start the Hayhooks server on `HAYHOOKS_HOST:HAYHOOKS_PORT`.

### Deploy a pipeline

Now, we will deploy a pipeline to chat with a website. We have created an example in the [examples/chat_with_website_streaming](examples/chat_with_website_streaming) folder.

In the example folder, we have two files:

- `chat_with_website.yml`: The pipeline definition in YAML format.
- `pipeline_wrapper.py` (mandatory): A pipeline wrapper that uses the pipeline definition.

#### Why a pipeline wrapper?

The pipeline wrapper provides a flexible foundation for deploying Haystack pipelines by allowing users to:

- Choose their preferred pipeline initialization method (YAML files, Haystack templates, or inline code)
- Define custom pipeline execution logic with configurable inputs and outputs
- Optionally expose OpenAI-compatible chat endpoints with streaming support for integration with interfaces like [open-webui](https://openwebui.com/)

The `pipeline_wrapper.py` file must contain an implementation of the `BasePipelineWrapper` class (see [here](src/hayhooks/server/utils/base_pipeline_wrapper.py) for more details).

A minimal `PipelineWrapper` looks like this:

```python
from pathlib import Path
from typing import List
from haystack import Pipeline
from hayhooks import BasePipelineWrapper

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: List[str], question: str) -> str:
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

#### run_api(...)

This method will be used to run the pipeline in API mode, when you call the `{pipeline_name}/run` endpoint.

**You can define the input arguments of the method according to your needs**.

```python
def run_api(self, urls: List[str], question: str, any_other_user_defined_argument: Any) -> str:
    ...
```

The input arguments will be used to generate a Pydantic model that will be used to validate the request body. The same will be done for the response type.

**NOTE**: Since Hayhooks will _dynamically_ create the Pydantic models, you need to make sure that the input arguments are JSON-serializable.

To deploy the pipeline, run:

```shell
hayhooks pipeline deploy-files -n chat_with_website examples/chat_with_website
```

This will deploy the pipeline with the name `chat_with_website`. Any error encountered during development will be printed to the console and show in the server logs.

#### PipelineWrapper development with overwrite option

During development, you can use the `--overwrite` flag to redeploy your pipeline without restarting the Hayhooks server:

```shell
hayhooks pipeline deploy-files -n {pipeline_name} --overwrite {pipeline_dir}
```

This is particularly useful when:

- Iterating on your pipeline wrapper implementation
- Debugging pipeline setup issues
- Testing different pipeline configurations

The `--overwrite` flag will:

1. Remove the existing pipeline from the registry
2. Delete the pipeline files from disk
3. Deploy the new version of your pipeline

For even faster development iterations, you can combine `--overwrite` with `--skip-saving-files` to avoid writing files to disk:

```shell
hayhooks pipeline deploy-files -n {pipeline_name} --overwrite --skip-saving-files {pipeline_dir}
```

This is useful when:

- You're making frequent changes during development
- You want to test a pipeline without persisting it
- You're running in an environment with limited disk access

#### Additional dependencies

After installing the Hayhooks package, it might happen that during pipeline deployment you need to install additional dependencies in order to correctly initialize the pipeline instance when calling the wrapper's `setup()` method. For instance, the `chat_with_website` pipeline requires the `trafilatura` package, which is **not installed by default**.

⚠️ Sometimes you may need to enable tracebacks in hayhooks to see the full error message. You can do this by setting the `HAYHOOKS_SHOW_TRACEBACKS` environment variable to `true` or `1`.

Then, assuming you've installed the Hayhooks package in a virtual environment, you will need to install the additional required dependencies yourself by running:

```shell
pip install trafilatura
```

### OpenAI-compatible endpoints generation

Hayhooks now can automatically generate OpenAI-compatible endpoints if you implement the `run_chat_completion` method in your pipeline wrapper.

This will make Hayhooks compatible with fully-featured chat interfaces like [open-webui](https://openwebui.com/), so you can use it as a backend for your chat interface.

### Using Hayhooks as `open-webui` backend

Requirements:

- Ensure you have [open-webui](https://openwebui.com/) up and running (you can do it easily using `docker`, check [their quick start guide](https://docs.openwebui.com/getting-started/quick-start)).
- Ensure you have Hayhooks server running somewhere. We will run it locally on `http://localhost:1416`.

#### Configuring `open-webui`

First, you need to **turn off `tags` and `title` generation from `Admin settings -> Interface`**:

![open-webui-settings](./docs/assets/open-webui-settings.png)

Then you have two options to connect Hayhooks as a backend.

Add a **Direct Connection** from `Settings -> Connections`:

NOTE: **Fill a random value as API key as it's not needed**

![open-webui-settings-connections](./docs/assets/open-webui-settings-connections.png)

Alternatively, you can add an additional **OpenAI API Connections** from `Admin settings -> Connections`:

![open-webui-admin-settings-connections](./docs/assets/open-webui-admin-settings-connections.png)

Even in this case, remember to **Fill a random value as API key**.

#### run_chat_completion(...)

To enable the automatic generation of OpenAI-compatible endpoints, you need only to implement the `run_chat_completion` method in your pipeline wrapper.

```python
def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
    ...
```

Let's update the previous example to add a streaming response:

```python
from pathlib import Path
from typing import Generator, List, Union
from haystack import Pipeline
from hayhooks import get_last_user_message, BasePipelineWrapper, log


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
- `messages`: The list of messages from the chat in the OpenAI format.
- `body`: The full body of the request.

Some notes:

- Since we have only the user messages as input here, the `question` is extracted from the last user message and the `urls` argument is hardcoded.
- In this example, the `run_chat_completion` method is returning a string, so the `open-webui` will receive a string as response and show the pipeline output in the chat all at once.
- The `body` argument contains the full request body, which may be used to extract more information like the `temperature` or the `max_tokens` (see the [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create) for more information).

Finally, to use non-streaming responses in `open-webui` you need also to turn of `Stream Chat Response` chat settings.

Here's a video example:

![chat-completion-example](./docs/assets/chat-completion.gif)

### Streaming responses in OpenAI-compatible endpoints

Hayhooks now provides a `streaming_generator` utility function that can be used to stream the pipeline output to the client.

Let's update the `run_chat_completion` method of the previous example:

```python
from pathlib import Path
from typing import Generator, List, Union
from haystack import Pipeline
from hayhooks import get_last_user_message, BasePipelineWrapper, log, streaming_generator


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

Now, if you run the pipeline and call one of the following endpoints:

- `{pipeline_name}/chat`
- `/chat/completions`
- `/v1/chat/completions`

You will see the pipeline output being streamed [in OpenAI-compatible format](https://platform.openai.com/docs/api-reference/chat/streaming) to the client and you'll be able to see the output in chunks.

Since output will be streamed to `open-webui` there's **no need to change `Stream Chat Response`** chat setting (leave it as `Default` or `On`).

Here's a video example:

![chat-completion-streaming-example](./docs/assets/chat-completion-streaming.gif)

### Integration with haystack OpenAIChatGenerator

Since Hayhooks is OpenAI-compatible, it can be used as a backend for the [haystack OpenAIChatGenerator](https://docs.haystack.deepset.ai/docs/openaichatgenerator).

Assuming you have a Haystack pipeline named `chat_with_website_streaming` and you have deployed it using Hayhooks, here's an example script of how to use it with the `OpenAIChatGenerator`:

```python
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.components.generators.utils import print_streaming_chunk

client = OpenAIChatGenerator(
    model="chat_with_website_streaming",
    api_key=Secret.from_token("not-relevant"),  # This is not used, you can set it to anything
    api_base_url="http://localhost:1416/v1/",
    streaming_callback=print_streaming_chunk,
)

client.run([ChatMessage.from_user("Where are the offices or SSI?")])
# > The offices of Safe Superintelligence Inc. (SSI) are located in Palo Alto, California, and Tel Aviv, Israel.

# > {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text='The offices of Safe >Superintelligence Inc. (SSI) are located in Palo Alto, California, and Tel Aviv, Israel.')], _name=None, _meta={'model': >'chat_with_website_streaming', 'index': 0, 'finish_reason': 'stop', 'completion_start_time': '2025-02-11T15:31:44.599726', >'usage': {}})]}
```

## Advanced usage 🔧

### Run Hayhooks programmatically

A Hayhooks app instance can be run programmatically created by using the `create_app` function. This is useful if you want to add custom routes or middleware to Hayhooks.

Here's an example script:

```python
import uvicorn
from hayhooks.settings import settings
from fastapi import Request
from hayhooks import create_app

# Create the Hayhooks app
hayhooks = create_app()


# Add a custom route
@hayhooks.get("/custom")
async def custom_route():
    return {"message": "Hi, this is a custom route!"}


# Add a custom middleware
@hayhooks.middleware("http")
async def custom_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Custom-Header"] = "custom-header-value"
    return response


if __name__ == "__main__":
    uvicorn.run("app:hayhooks", host=settings.host, port=settings.port)
```

### Deployment guidelines 📦

For detailed deployment guidelines, see [deployment_guidelines.md](docs/deployment_guidelines.md).

### Legacy Features

#### Deploy a pipeline using only its YAML definition

**⚠️ This way of deployment is not maintained anymore and will be deprecated in the future**.

We're still supporting the Hayhooks _former_ way to deploy a pipeline.

The former command `hayhooks deploy` is now changed to `hayhooks pipeline deploy` and can be used to deploy a pipeline only from a YAML definition file.

For example:

```shell
hayhooks pipeline deploy -n chat_with_website examples/chat_with_website/chat_with_website.yml
```

This will deploy the pipeline with the name `chat_with_website` from the YAML definition file `examples/chat_with_website/chat_with_website.yml`. You then can check the generated docs at `http://HAYHOOKS_HOST:HAYHOOKS_PORT/docs` or `http://HAYHOOKS_HOST:HAYHOOKS_PORT/redoc`, looking at the `POST /chat_with_website` endpoint.

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
