# RAG example: indexing and querying with Elasticsearch

This example will show how you can deploy with Hayhooks an index pipeline and a query pipeline, using Elasticsearch as the document store.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)

## 1. Create a virtual environment

It's always a good idea to create a virtual environment to install the dependencies in.

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install the dependencies

We'll need to install the Hayhooks package and the other dependencies needed for the pipelines.

```bash
pip install -r requirements.txt
```

## 3a. Launch Hayhooks

For simplicity, we'll use the Hayhooks CLI to launch the Hayhooks server.

```bash
hayhooks run
```

You can achieve the same result by running [a Docker image with Hayhooks already installed](https://github.com/deepset-ai/hayhooks-open-webui-docker-compose).

## 3b. Launch Elasticsearch

```bash
docker compose up
```

This will start an [Elasticsearch](https://www.elastic.co/elasticsearch) instance on port 9200.

## 4. Deploy the pipelines

```bash
hayhooks pipeline deploy-files -n indexing indexing_pipeline
hayhooks pipeline deploy-files -n query query_pipeline
```

Let's also check on <http://localhost:1416/docs> if the pipelines are deployed correctly.

## 5. Test the indexing pipeline

We'll use `hayhooks pipeline run` to run the indexing pipeline.

We will index the files in the `files_to_index` directory, then launch a query to check if the indexing was successful.

```bash
hayhooks pipeline run indexing --dir files_to_index
```

## 6. Test the query pipeline

```bash
hayhooks pipeline run query --param 'question="is this recipe vegan?"'
```
