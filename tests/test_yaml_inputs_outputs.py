import re
from pathlib import Path
from typing import Any

import pytest
from haystack.dataclasses import ChatMessage

from hayhooks.server.exceptions import InvalidYamlIOError
from hayhooks.server.utils.yaml_utils import (
    InputResolution,
    OutputResolution,
    get_components_from_outputs,
    get_inputs_outputs_from_yaml,
    get_streaming_components_from_yaml,
)


def test_get_inputs_outputs_from_yaml_matches_pipeline_metadata():
    yaml_path = Path(__file__).parent / "test_files" / "yaml" / "inputs_outputs_pipeline.yml"
    yaml_source = yaml_path.read_text()

    result = get_inputs_outputs_from_yaml(yaml_source)

    assert set(result.keys()) == {"inputs", "outputs"}
    assert set(result["inputs"].keys()) == {"urls", "query"}
    assert set(result["outputs"].keys()) == {"replies"}

    assert isinstance(result["inputs"]["urls"], InputResolution)
    assert result["inputs"]["urls"].path == "fetcher.urls"
    assert result["inputs"]["urls"].component == "fetcher"
    assert result["inputs"]["urls"].name == "urls"
    assert result["inputs"]["urls"].type == list[str]
    assert result["inputs"]["urls"].required is True

    assert isinstance(result["inputs"]["query"], InputResolution)
    assert result["inputs"]["query"].path == "prompt.query"
    assert result["inputs"]["query"].component == "prompt"
    assert result["inputs"]["query"].name == "query"
    assert result["inputs"]["query"].type == Any

    assert isinstance(result["outputs"]["replies"], OutputResolution)
    assert result["outputs"]["replies"].path == "llm.replies"
    assert result["outputs"]["replies"].component == "llm"
    assert result["outputs"]["replies"].name == "replies"
    assert result["outputs"]["replies"].type == list[ChatMessage]


def test_get_inputs_outputs_from_yaml_raises_when_missing_inputs_outputs():
    yaml_path = Path(__file__).parent / "test_files" / "mixed" / "chat_with_website" / "chat_with_website.yml"
    yaml_source = yaml_path.read_text()

    with pytest.raises(
        InvalidYamlIOError, match=re.escape("YAML pipeline must declare at least one of 'inputs' or 'outputs'.")
    ):
        get_inputs_outputs_from_yaml(yaml_source)


def test_get_streaming_components_from_yaml_with_valid_config():
    yaml_source = """
components:
  prompt_1:
    type: haystack.components.builders.chat_prompt_builder.ChatPromptBuilder
    init_parameters:
      template: |
        {% message role="user" %}
        {{query}}
        {% endmessage %}
      required_variables: "*"
  llm_1:
    type: haystack.components.generators.chat.openai.OpenAIChatGenerator
    init_parameters:
      api_key:
        env_vars:
          - OPENAI_API_KEY
        strict: true
        type: env_var
      model: gpt-4o-mini
      generation_kwargs: {}
  llm_2:
    type: haystack.components.generators.chat.openai.OpenAIChatGenerator
    init_parameters:
      api_key:
        env_vars:
          - OPENAI_API_KEY
        strict: true
        type: env_var
      model: gpt-4o-mini
      generation_kwargs: {}

connections:
  - sender: prompt_1.prompt
    receiver: llm_1.messages
  - sender: llm_1.replies
    receiver: llm_2.messages

inputs:
  query: prompt_1.query

outputs:
  replies: llm_2.replies

streaming_components:
  - llm_1
"""
    result = get_streaming_components_from_yaml(yaml_source)

    assert result is not None
    assert result == ["llm_1"]


def test_get_streaming_components_from_yaml_without_config():
    yaml_source = """
components:
  prompt:
    type: haystack.components.builders.chat_prompt_builder.ChatPromptBuilder
    init_parameters:
      template: |
        {% message role="user" %}
        {{query}}
        {% endmessage %}
      required_variables: "*"
  llm:
    type: haystack.components.generators.chat.openai.OpenAIChatGenerator
    init_parameters:
      api_key:
        env_vars:
          - OPENAI_API_KEY
        strict: true
        type: env_var
      model: gpt-4o-mini
      generation_kwargs: {}

connections:
  - sender: prompt.prompt
    receiver: llm.messages

inputs:
  query: prompt.query

outputs:
  replies: llm.replies
"""
    result = get_streaming_components_from_yaml(yaml_source)

    assert result is None


def test_get_streaming_components_from_yaml_with_invalid_type():
    yaml_source = """
components:
  prompt:
    type: haystack.components.builders.chat_prompt_builder.ChatPromptBuilder
    init_parameters:
      template: |
        {% message role="user" %}
        {{query}}
        {% endmessage %}
      required_variables: "*"
  llm:
    type: haystack.components.generators.chat.openai.OpenAIChatGenerator
    init_parameters:
      api_key:
        env_vars:
          - OPENAI_API_KEY
        strict: true
        type: env_var
      model: gpt-4o-mini
      generation_kwargs: {}

connections:
  - sender: prompt.prompt
    receiver: llm.messages

inputs:
  query: prompt.query

outputs:
  replies: llm.replies

streaming_components: 123
"""
    result = get_streaming_components_from_yaml(yaml_source)

    assert result is None


def test_get_streaming_components_from_yaml_converts_to_str():
    yaml_source = """
components:
  prompt:
    type: haystack.components.builders.chat_prompt_builder.ChatPromptBuilder
    init_parameters:
      template: |
        {% message role="user" %}
        {{query}}
        {% endmessage %}
      required_variables: "*"
  llm_1:
    type: haystack.components.generators.chat.openai.OpenAIChatGenerator
    init_parameters:
      api_key:
        env_vars:
          - OPENAI_API_KEY
        strict: true
        type: env_var
      model: gpt-4o-mini
      generation_kwargs: {}
  llm_2:
    type: haystack.components.generators.chat.openai.OpenAIChatGenerator
    init_parameters:
      api_key:
        env_vars:
          - OPENAI_API_KEY
        strict: true
        type: env_var
      model: gpt-4o-mini
      generation_kwargs: {}

connections:
  - sender: prompt.prompt
    receiver: llm_1.messages
  - sender: llm_1.replies
    receiver: llm_2.messages

inputs:
  query: prompt.query

outputs:
  replies: llm_2.replies

streaming_components:
  - llm_1
  - llm_2
"""
    result = get_streaming_components_from_yaml(yaml_source)

    assert result is not None
    assert result == ["llm_1", "llm_2"]
    assert all(isinstance(item, str) for item in result)


def test_get_streaming_components_from_yaml_with_all_keyword():
    yaml_source = """
components:
  prompt:
    type: haystack.components.builders.chat_prompt_builder.ChatPromptBuilder
    init_parameters:
      template: |
        {% message role="user" %}
        {{query}}
        {% endmessage %}
      required_variables: "*"
  llm_1:
    type: haystack.components.generators.chat.openai.OpenAIChatGenerator
    init_parameters:
      api_key:
        env_vars:
          - OPENAI_API_KEY
        strict: true
        type: env_var
      model: gpt-4o-mini
      generation_kwargs: {}
  llm_2:
    type: haystack.components.generators.chat.openai.OpenAIChatGenerator
    init_parameters:
      api_key:
        env_vars:
          - OPENAI_API_KEY
        strict: true
        type: env_var
      model: gpt-4o-mini
      generation_kwargs: {}

connections:
  - sender: prompt.prompt
    receiver: llm_1.messages
  - sender: llm_1.replies
    receiver: llm_2.messages

inputs:
  query: prompt.query

outputs:
  replies: llm_2.replies

streaming_components: all
"""
    result = get_streaming_components_from_yaml(yaml_source)

    assert result == "all"


def test_get_components_from_outputs():
    yaml_path = Path(__file__).parent / "test_files" / "yaml" / "inputs_outputs_pipeline.yml"
    yaml_source = yaml_path.read_text()

    result = get_inputs_outputs_from_yaml(yaml_source)
    components = get_components_from_outputs(result["outputs"])

    assert components == {"llm"}


def test_get_components_from_outputs_multiple_components():
    yaml_path = Path(__file__).parent / "test_files" / "yaml" / "multi_output_pipeline.yml"
    yaml_source = yaml_path.read_text()

    result = get_inputs_outputs_from_yaml(yaml_source)
    components = get_components_from_outputs(result["outputs"])

    assert isinstance(components, set)
    assert all(isinstance(component, str) for component in components)
    assert components == {"double", "second_addition"}
    assert "double.value" not in components
    assert "second_addition.result" not in components


def test_get_components_from_outputs_empty():
    assert get_components_from_outputs({}) == set()
