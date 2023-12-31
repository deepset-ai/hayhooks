# Hayhooks

[![PyPI - Version](https://img.shields.io/pypi/v/hayhooks.svg)](https://pypi.org/project/hayhooks)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hayhooks.svg)](https://pypi.org/project/hayhooks)

-----

**Table of Contents**

- [Hayhooks](#hayhooks)
  - [Quick start](#quick-start)
    - [Install the package](#install-the-package)
    - [Check Hayhooks status](#check-hayhooks-status)
    - [Deploy a Haystack pipeline](#deploy-a-haystack-pipeline)
    - [Have a look at the API schema](#have-a-look-at-the-api-schema)
  - [License](#license)

## Quick start

### Install the package

Start by installing the package:

```console
pip install hayhooks
```

The `hayhooks` package ships both the server and the client component, and the client is capable of starting the
server. From a shell, start the server with:

```console
$ hayhooks run
INFO:     Started server process [44782]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:1416 (Press CTRL+C to quit)
```
### Check Hayhooks status

From a different shell, you can query the status of the server with:

```console
$ hayhooks status
Hayhooks server is up and running.
```
### Deploy a Haystack pipeline

Time to deploy a Haystack pipeline. The pipeline must be in Yaml format (the output of [`pipeline.dump()`](https://docs.haystack.deepset.ai/v2.0/docs/serialization#converting-a-pipeline-to-yaml)), if you don't have one at hand, you can use
one from this repository. From the root of the repo:

```console
$ hayhooks deploy tests/test_files/test_pipeline_01.yml
Pipeline successfully deployed with name: test_pipeline_01
```

Another call to `status` should confirm your pipeline is ready to serve requests:

```console
$ hayhooks status
Hayhooks server is up and running.

Pipelines deployed:
- test_pipeline_01
```

### Have a look at the API schema

Hayhooks will use introspection to set up the OpenAPI schema accordingly to the inputs and outputs of your pipeline,
and to see how this works let's get the pipeline diagram with:

```console
$ curl http://localhost:1416/draw/test_pipeline_01 --output test_pipeline_01.png
```

The downloaded image should look like this:

![test pipeline](img/test_pipeline_01.png)

As you can see, the pipeline in order to start requires an input of type `int` named `value`, and optionally we can
pass another input of type `int` named `add`. At the end of the run, the pipeline will return an output of type `int`
named `result`.

If you open a browser at [http://localhost:1416/docs#/](http://localhost:1416/docs#/) you should see two schemas, one
for the Request, where we'll pass the pipeline inputs (note how `add` is optional):

```
Test_pipeline_01RunRequest
    first_addition
        value* integer
        add (integer | null)
```

And another one for the Response, where we'll receive the pipeline results:

```
Test_pipeline_01RunResponse
    double
        value* integer
```


## License

`hayhooks` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
