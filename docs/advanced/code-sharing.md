# Sharing code between pipeline wrappers

Keep shared logic in a separate folder and add it to the Hayhooks Python Path so your wrappers can import it directly. This page mirrors the guidance in the README and the example in `examples/shared_code_between_wrappers`.

## Add a folder to Hayhooks Python Path

You can point Hayhooks to a folder containing shared code using any of the following. Both absolute and relative paths are supported (relative paths are resolved from the current working directory).

### Option 1: Environment variable

```bash
export HAYHOOKS_ADDITIONAL_PYTHON_PATH='./common'
hayhooks run
```

### Option 2: .env file

```bash
echo "HAYHOOKS_ADDITIONAL_PYTHON_PATH='./common'" >> .env
hayhooks run
```

### Option 3: CLI flag

```bash
hayhooks run --additional-python-path ./common
```

Once set, code inside that folder can be imported in your wrappers. For example, assuming you have a `my_custom_lib.py` module in the `common` folder which contains the `sum_two_numbers` function, you can import it in your wrappers like this:

```python
from my_custom_lib import sum_two_numbers
```

## Quick start with the included example

The repository includes a minimal, working example: [examples/shared_code_between_wrappers/README.md](https://github.com/deepset-ai/hayhooks/blob/main/examples/shared_code_between_wrappers/README.md).
