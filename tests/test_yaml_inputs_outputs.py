import re
from pathlib import Path
from typing import Any

import pytest

from hayhooks.server.exceptions import InvalidYamlIOError
from hayhooks.server.utils.yaml_utils import InputResolution, OutputResolution, get_inputs_outputs_from_yaml


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
    assert result["outputs"]["replies"].type == list[str]


def test_get_inputs_outputs_from_yaml_raises_when_missing_inputs_outputs():
    yaml_path = Path(__file__).parent / "test_files" / "mixed" / "chat_with_website" / "chat_with_website.yml"
    yaml_source = yaml_path.read_text()

    with pytest.raises(
        InvalidYamlIOError, match=re.escape("YAML pipeline must declare at least one of 'inputs' or 'outputs'.")
    ):
        get_inputs_outputs_from_yaml(yaml_source)
