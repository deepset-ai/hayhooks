# RAG System Example

Build a Retrieval-Augmented Generation flow: ingest documents, embed and store them, retrieve by similarity, and answer questions with an LLM.

## Where is the code?

- End-to-end example: [examples/rag_indexing_query](https://github.com/deepset-ai/hayhooks/tree/main/examples/rag_indexing_query)
  - `indexing_pipeline/` - Handles document upload and indexing
  - `query_pipeline/` - Retrieves and generates answers

## Quick Start (from repository root)

```bash
# 1) Enter the example
cd examples/rag_indexing_query

# 2) (Recommended) Create and activate a virtual env, then install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) Launch Hayhooks (in a separate terminal if you prefer)
hayhooks run

# 4) Launch Elasticsearch
docker compose up

# 5) Deploy the pipelines
hayhooks pipeline deploy-files -n indexing indexing_pipeline
hayhooks pipeline deploy-files -n query query_pipeline

# 6) Index sample files
hayhooks pipeline run indexing --dir files_to_index

# 7) Ask a question
hayhooks pipeline run query --param 'question="is this recipe vegan?"'

# Optional: check API docs
# http://localhost:1416/docs
```

!!! info "Additional Information"
    - See [File Upload Support](../features/file-upload-support.md) for wrapper signature and request format
    - Choose appropriate embedding models and document stores for your scale

## Related

- General guide: [Main docs](../index.md)
- Examples index: [Examples Overview](overview.md)
