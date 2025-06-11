# Sharing code between pipeline wrappers

This example shows how to share code between pipeline wrappers using `HAYHOOKS_ADDITIONAL_PYTHON_PATH`.

**NOTE**: We will run all the commands from the example directory, so `./examples/shared_code_between_wrappers`.

## 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install Hayhooks

```bash
pip install hayhooks
```

## 3. Create the common code

In this example, we'll add the code from `common` folder to the Hayhooks [Python Path](https://docs.python.org/3/library/sys_path_init.html).
You have three ways to do this:

### Option 1: Set the environment variable `HAYHOOKS_ADDITIONAL_PYTHON_PATH` to the path of the `common` folder

You can use both absolute and relative paths. Since we are running all the commands from the example directory, we'll use a relative path.

```bash
export HAYHOOKS_ADDITIONAL_PYTHON_PATH='./common'
```

And then launch Hayhooks as usual:

```bash
hayhooks run
```

### Option 2: Add `HAYHOOKS_ADDITIONAL_PYTHON_PATH` to the `.env` file

```bash
echo "HAYHOOKS_ADDITIONAL_PYTHON_PATH='./common'" >> .env
```

And then Launch Hayhooks as usual:

```bash
hayhooks run
```

### Option 3: Launch Hayhooks with the `--additional-python-path` flag

```bash
hayhooks run --additional-python-path ./common
```

In both cases, Hayhooks will add the `common` folder to the Python Path. This means you can import the code from the `common` folder in your pipeline wrappers.

For example, in `pipeline_1/pipeline_wrapper.py` we have:

```python
from my_custom_lib import sum_two_numbers
```

## 4. Deploy the pipelines

Both `pipeline_1` and `pipeline_2` will use the code from the `common` folder.

```bash
hayhooks pipeline deploy-files -n pipeline1 input_pipelines/pipeline_1
hayhooks pipeline deploy-files -n pipeline2 input_pipelines/pipeline_2
```

This will create a `pipelines` folder in the current working directory, with the pipeline wrappers for `pipeline_1` and `pipeline_2`.

## 5. Test the pipelines

```bash
curl -X 'POST' \
  'http://localhost:1416/pipeline1/run' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "a": 1,
  "b": 3
}'
```

Will output:

```json
{"result": 4}
```
