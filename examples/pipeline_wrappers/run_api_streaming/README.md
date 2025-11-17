# Run API Streaming Example

This example shows how to stream tokens directly from `/<pipeline>/run` endpoint
using `streaming_generator()`. Instead of waiting for the pipeline to finish and returning
the final string, the wrapper yields streaming chunks as soon as the underlying LLM produces them.

## Highlights

- Implements `run_api()` to return a generator of `StreamingChunk` objects.
- Uses the helper `streaming_generator()` from Hayhooks (no manual queue management required).
- Demonstrates how `/run` automatically becomes a `StreamingResponse` after returning a generator.

## Deploy & Try It

```bash
# 1. Set your OpenAI API key (the pipeline uses OpenAIChatGenerator)
export OPENAI_API_KEY=your_api_key_here

# 2. Deploy the example
hayhooks deploy examples/pipeline_wrappers/run_api_streaming

# 3. Call the /run endpoint with streaming enabled (-N keeps curl open)
curl -N http://localhost:1416/run_api_streaming/run \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Redis?", "urls": ["https://www.redis.io"]}'
```

You should see words streaming in the terminal instead of receiving the entire answer at once.

## How It Works

```python
def run_api(self, question: str, urls: Optional[list[str]] = None) -> Generator[StreamingChunk, None, None]:
    return streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={
            "fetcher": {"urls": urls or DEFAULT_URLS},
            "prompt": {"query": question},
        },
    )
```

By returning the generator, Hayhooks automatically wraps the response in a `StreamingResponse`
and takes care of cleaning up the generator once streaming is complete.

> **Note:** For async pipelines, use `async_streaming_generator()` inside `run_api_async()` instead.
> The async version works identically but returns an async generator that Hayhooks will handle automatically.
