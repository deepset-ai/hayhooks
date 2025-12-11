# Relative Imports in Pipeline Wrappers

This example demonstrates how to use **relative imports** to organize your pipeline wrapper code into multiple files.

## Structure

```text
relative_imports/
├── pipeline_wrapper.py    # Main wrapper with PipelineWrapper class
├── utils.py               # Helper functions
└── README.md              # This file
```

## Key Features

- **No configuration needed** - Relative imports work out of the box
- **Clean organization** - Split complex logic into separate modules
- **IDE support** - Full autocomplete and type checking
- **Tracing compatible** - Works with Phoenix/OpenInference and other tracing libraries

## How It Works

Pipeline wrappers are loaded as Python packages, which enables relative imports:

```python
# In pipeline_wrapper.py
from .utils import greet, calculate_sum  # ← Relative import
```

## Ruff Linting Note

If your project uses [ruff](https://docs.astral.sh/ruff/) with the `flake8-tidy-imports` plugin, you may need to disable the [`TID252`](https://docs.astral.sh/ruff/rules/relative-imports/) rule which flags relative imports.

Add this comment at the top of your `pipeline_wrapper.py`:

```python
# ruff: noqa: TID252
```

Or configure it in your `pyproject.toml`:

```toml
[tool.ruff.lint.flake8-tidy-imports]
ban-relative-imports = "parents"  # Allow sibling relative imports
```

## Running the Example

### 1. Start Hayhooks

```bash
hayhooks run
```

### 2. Deploy the Pipeline

```bash
hayhooks pipeline deploy-files -n calculator examples/pipeline_wrappers/relative_imports
```

### 3. Test the Pipeline

```bash
curl -X 'POST' \
  'http://localhost:1416/calculator/run' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Alice",
    "numbers": [1, 2, 3, 4, 5]
  }'
```

**Expected output:**

```json
{
  "result": {
    "greeting": "Hello, Alice!",
    "sum": 15,
    "average": 3.0
  }
}
```

## When to Use Relative Imports

- Splitting a large wrapper into logical modules
- Keeping helper functions close to where they're used
- Organizing prompts, configs, or utilities separately
- Any code that's specific to a single pipeline

For sharing code across **multiple pipelines**, see the [shared_code_between_wrappers](../../../shared_code_between_wrappers/) example.
