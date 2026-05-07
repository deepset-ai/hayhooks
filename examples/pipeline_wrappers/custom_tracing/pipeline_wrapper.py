"""
Pipeline wrapper that classifies Haystack pipeline YAML as I/O-bound or CPU-bound.

The classifier is a custom Haystack component that performs a direct OpenAI API
call and emits custom tracing spans for each stage.
"""

import json
import os
from pathlib import Path
from string import Template
from typing import Any

from fastapi import HTTPException, UploadFile
from haystack.core.component import component
from openai import OpenAI

from hayhooks import BasePipelineWrapper
from hayhooks.server.tracing import build_trace_tags, trace_operation

EXAMPLE_PIPELINE_NAME = "custom_tracing"
PROMPT_TEMPLATE_FILE = "classification_prompt_template.txt"

DEFAULT_MODEL = "gpt-5.4-mini"

SPAN_CLASSIFIER_COMPONENT = "haystack.component.run"
SPAN_RESOLVE_INPUT = "hayhooks.classifier.resolve_input"
SPAN_RENDER_PROMPT = "hayhooks.classifier.render_prompt"
SPAN_OPENAI_CALL = "hayhooks.classifier.openai_call"


def _resolve_yaml_input(
    pipeline_yaml: str | None,
    files: list[UploadFile] | None,
) -> tuple[str, str, str]:
    inline_yaml = (pipeline_yaml or "").strip()
    if inline_yaml:
        return inline_yaml, "inline", ""

    if files:
        uploaded_file = files[0]
        uploaded_yaml = uploaded_file.file.read().decode("utf-8", errors="ignore").strip()
        if uploaded_yaml:
            return uploaded_yaml, "file_upload", uploaded_file.filename or "uploaded_file"

    raise HTTPException(status_code=400, detail="Provide pipeline_yaml or upload a YAML file.")


@component
class PipelineBoundClassifier:
    def __init__(self, prompt_template: str):
        self._prompt_template = Template(prompt_template)

    @component.output_types(
        classification=str,
        confidence=float,
        rationale=str,
        model=str,
    )
    def run(self, pipeline_yaml: str) -> dict[str, Any]:
        tags = {
            "hayhooks.pipeline.name": EXAMPLE_PIPELINE_NAME,
            "haystack.component.name": "PipelineBoundClassifier",
        }
        with trace_operation(SPAN_CLASSIFIER_COMPONENT, tags=build_trace_tags(tags)):
            model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="OPENAI_API_KEY must be set.")

            with trace_operation(SPAN_RENDER_PROMPT, tags=build_trace_tags(tags)):
                prompt = self._prompt_template.safe_substitute(
                    pipeline_yaml=pipeline_yaml.strip(),
                )

            with trace_operation(
                SPAN_OPENAI_CALL,
                tags=build_trace_tags(tags, **{"hayhooks.classifier.llm.model": model}),
            ):
                response_text = self._call_openai(prompt=prompt, model=model, api_key=api_key)

            try:
                result = json.loads(response_text)
            except (json.JSONDecodeError, TypeError) as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Could not parse model output as JSON: {exc}",
                ) from exc

            return {
                "classification": str(result.get("classification", "")),
                "confidence": float(result.get("confidence", 0.0)),
                "rationale": str(result.get("rationale", "")),
                "model": model,
            }

    @staticmethod
    def _call_openai(*, prompt: str, model: str, api_key: str) -> str:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI API request failed: {exc}") from exc

        choices = response.choices
        if not choices:
            return ""
        return (choices[0].message.content or "").strip()


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        prompt_path = Path(__file__).parent / PROMPT_TEMPLATE_FILE
        self.classifier = PipelineBoundClassifier(prompt_template=prompt_path.read_text())

    def run_api(
        self,
        pipeline_yaml: str | None = None,
        files: list[UploadFile] | None = None,
    ) -> dict[str, Any]:
        with trace_operation(
            SPAN_RESOLVE_INPUT,
            tags=build_trace_tags(
                {
                    "hayhooks.pipeline.name": EXAMPLE_PIPELINE_NAME,
                    "hayhooks.classifier.has_inline_yaml": bool(pipeline_yaml and pipeline_yaml.strip()),
                    "hayhooks.classifier.uploaded_file_count": len(files or []),
                }
            ),
        ):
            normalized_yaml, input_source, input_filename = _resolve_yaml_input(pipeline_yaml, files)

        result = self.classifier.run(pipeline_yaml=normalized_yaml)
        result["input_source"] = input_source
        result["input_filename"] = input_filename
        return result
