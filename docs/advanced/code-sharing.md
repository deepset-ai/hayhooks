# Code Sharing Between Wrappers

Share common code between pipeline wrappers by adding a folder to the Hayhooks Python path.

## Configuration

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

## Usage

Once configured, import shared code in your wrappers:

```python
# In your pipeline_wrapper.py
from my_custom_lib import sum_two_numbers

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, a: int, b: int) -> int:
        return sum_two_numbers(a, b)
```

## Example

See [examples/shared_code_between_wrappers](https://github.com/deepset-ai/hayhooks/tree/main/examples/shared_code_between_wrappers) for a complete working example.
