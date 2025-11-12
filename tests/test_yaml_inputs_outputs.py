import re
from pathlib import Path
from typing import Any

import pytest
from haystack.dataclasses import ChatMessage

from hayhooks.server.exceptions import InvalidYamlIOError
from hayhooks.server.utils.deploy_utils import map_flat_inputs_to_components
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
    assert result["inputs"]["urls"].targets == ["fetcher.urls"]
    assert result["inputs"]["urls"].required is True

    assert isinstance(result["inputs"]["query"], InputResolution)
    assert result["inputs"]["query"].path == "prompt.query"
    assert result["inputs"]["query"].component == "prompt"
    assert result["inputs"]["query"].name == "query"
    assert result["inputs"]["query"].type == Any
    assert result["inputs"]["query"].targets == ["prompt.query"]
    assert result["inputs"]["query"].required is True

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


def test_get_inputs_outputs_from_yaml_handles_list_declared_inputs():
    yaml_path = Path(__file__).parent / "test_files" / "yaml" / "list_input.yml"
    yaml_source = yaml_path.read_text()

    result = get_inputs_outputs_from_yaml(yaml_source)

    assert set(result.keys()) == {"inputs", "outputs"}
    assert set(result["inputs"].keys()) == {"query"}
    assert set(result["outputs"].keys()) == {"answers"}

    query_input = result["inputs"]["query"]

    # Here we're testing that we take `chat_summary_prompt_builder.query`
    # as a reference component for detecting the type, then we will pass the `query_input` value
    # to both `chat_summary_prompt_builder.query` and `answer_builder.query`
    # Note that `query_input` is _always_ required, since it's present in the `inputs` section.
    assert query_input.path == "chat_summary_prompt_builder.query"
    assert query_input.component == "chat_summary_prompt_builder"
    assert query_input.name == "query"
    assert query_input.type == Any
    assert query_input.required is True
    assert query_input.targets == [
        "chat_summary_prompt_builder.query",
        "answer_builder.query",
    ]

    answers_output = result["outputs"]["answers"]
    assert answers_output.path == "answer_builder.answers"
    assert answers_output.component == "answer_builder"
    assert answers_output.name == "answers"


def test_get_inputs_outputs_from_yaml_raises_on_duplicate_input_targets():
    yaml_path = Path(__file__).parent / "test_files" / "yaml" / "broken" / "duplicate_input_target.yml"
    yaml_source = yaml_path.read_text()

    with pytest.raises(
        InvalidYamlIOError,
        match=re.escape(
            "Declared input 'another_input' targets 'chat_summary_prompt_builder.query'; "
            "'chat_summary_prompt_builder.query' already targeted by declared input 'query'. "
            "Each pipeline input target may be declared only once."
        ),
    ):
        get_inputs_outputs_from_yaml(yaml_source)


def test_map_flat_inputs_to_components_expands_targets():
    yaml_path = Path(__file__).parent / "test_files" / "yaml" / "list_input.yml"
    yaml_source = yaml_path.read_text()
    resolved = get_inputs_outputs_from_yaml(yaml_source)

    expanded = map_flat_inputs_to_components({"query": "value"}, resolved["inputs"])

    assert expanded == {
        "chat_summary_prompt_builder": {"query": "value"},
        "answer_builder": {"query": "value"},
    }
