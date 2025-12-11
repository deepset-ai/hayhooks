# Code Sharing Between Wrappers

Hayhooks provides two ways to organize and share code in pipeline wrappers:

1. **Relative Imports** - Import from sibling modules within the same pipeline folder
2. **Shared Python Path** - Share code across multiple pipeline wrappers

## Relative Imports (Recommended)

Pipeline wrappers are loaded as Python packages, enabling you to use **relative imports** to organize your code into multiple files within the same pipeline folder.

### Structure

```text
my_pipeline/
├── pipeline_wrapper.py    # Main wrapper
├── utils.py               # Helper functions
├── prompts.py             # Prompt templates
└── config.py              # Configuration
```

### Usage

Use Python's relative import syntax (`from .module import ...`):

```python
# pipeline_wrapper.py
from haystack import Pipeline
from hayhooks import BasePipelineWrapper

# Relative imports from sibling modules
from .utils import process_text, format_response
from .prompts import SYSTEM_PROMPT
from .config import DEFAULT_MODEL

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()
        # ... setup using imported utilities

    def run_api(self, query: str) -> str:
        processed = process_text(query)
        result = self.pipeline.run({"prompt": {"query": processed}})
        return format_response(result)
```

### Benefits

- **No configuration needed** - Works out of the box
- **Clean organization** - Split large wrappers into logical modules
- **IDE support** - Full autocomplete and type checking
- **Tracing compatibility** - Works with Phoenix/OpenInference and other tracing libraries

!!! note "Ruff Linting"
    If your project uses [ruff](https://docs.astral.sh/ruff/) with the `flake8-tidy-imports` plugin, you may need to disable the [`TID252`](https://docs.astral.sh/ruff/rules/relative-imports/) rule which bans relative imports. Add this comment at the top of your `pipeline_wrapper.py`:

    ```python
    # ruff: noqa: TID252
    ```

    Or configure it in your `pyproject.toml`:

    ```toml
    [tool.ruff.lint.flake8-tidy-imports]
    ban-relative-imports = "parents"  # Allow sibling relative imports
    ```

### Example

See [examples/pipeline_wrappers/relative_imports](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/relative_imports) for a complete working example.

---

## Shared Python Path

For sharing code **across multiple pipeline wrappers**, add a common folder to the Hayhooks Python path.

### Configuration

Set `HAYHOOKS_ADDITIONAL_PYTHON_PATH` to point to your shared code directory:

=== "Environment Variable"
    ```bash
    export HAYHOOKS_ADDITIONAL_PYTHON_PATH='./common'
    hayhooks run
    ```

=== ".env File"
    ```bash
    # .env
    HAYHOOKS_ADDITIONAL_PYTHON_PATH=./common
    ```

=== "CLI Flag"
    ```bash
    hayhooks run --additional-python-path ./common
    ```

### Usage

Once configured, import shared code in your wrappers:

```python
# In your pipeline_wrapper.py
from my_custom_lib import sum_two_numbers

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, a: int, b: int) -> int:
        return sum_two_numbers(a, b)
```

### Example

See [examples/shared_code_between_wrappers](https://github.com/deepset-ai/hayhooks/tree/main/examples/shared_code_between_wrappers) for a complete working example.

---

## Choosing the Right Approach

| Use Case | Recommended Approach |
|----------|---------------------|
| Splitting a large wrapper into modules | **Relative Imports** |
| Helpers specific to one pipeline | **Relative Imports** |
| Sharing utilities across many pipelines | **Shared Python Path** |
| Company-wide libraries | **Shared Python Path** |
