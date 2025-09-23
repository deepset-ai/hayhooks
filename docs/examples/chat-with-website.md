# Chat with Website Example

This example demonstrates how to create a pipeline that can chat with website content, fetching and analyzing web pages in real-time.

## Overview

The chat-with-website pipeline allows users to:

- Ask questions about specific websites
- Fetch and analyze web page content
- Provide conversational answers based on website content
- Handle multiple websites in a single conversation

## Pipeline Implementation

### Complete Pipeline Code

```python
# pipeline_wrapper.py
from typing import AsyncGenerator, List, Dict, Any
from haystack import Pipeline
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIChatGenerator
from haystack.dataclasses import Document
from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message
import re
import logging

logger = logging.getLogger(__name__)

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup the chat-with-website pipeline"""
        # Initialize components
        self.fetcher = LinkContentFetcher()
        self.converter = HTMLToDocument()
        self.cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=True
        )
        self.prompt_builder = PromptBuilder(
            template="""You are a helpful assistant that answers questions based on website content.

Website: {{website_url}}
Content: {{documents}}

User Question: {{query}}

Please provide a helpful answer based on the website content above. If the content doesn't contain enough information to answer the question, please say so.

Answer:"""
        )
        self.llm = OpenAIChatGenerator(
            model="gpt-4",
            streaming_callback=lambda x: None
        )

        # Build pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("fetcher", self.fetcher)
        self.pipeline.add_component("converter", self.converter)
        self.pipeline.add_component("cleaner", self.cleaner)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)

        # Connect components
        self.pipeline.connect("fetcher.content", "converter")
        self.pipeline.connect("converter.documents", "cleaner")
        self.pipeline.connect("cleaner.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from user message"""
        # Simple URL extraction
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        """Run chat completion with website content"""
        question = get_last_user_message(messages)

        # Extract URLs from user message
        urls = self.extract_urls(question)

        if not urls:
            # If no URLs provided, ask for them
            yield "Please provide a website URL to chat about. For example: 'Tell me about https://example.com'"
            return

        try:
            # Process each URL
            for url in urls:
                logger.info(f"Processing URL: {url}")

                # Fetch and process the website
                result = async_streaming_generator(
                    pipeline=self.pipeline,
                    pipeline_run_args={
                        "fetcher": {"urls": [url]},
                        "prompt_builder": {
                            "query": question,
                            "website_url": url
                        }
                    },
                )

                # Stream the response
                async for chunk in result:
                    yield chunk

        except Exception as e:
            logger.error(f"Error processing website: {str(e)}")
            yield f"Sorry, I encountered an error while processing the website: {str(e)}"

    def run_api(self, query: str = "", urls: List[str] = None) -> str:
        """Run API endpoint for website chat"""
        if not urls:
            return "Please provide at least one URL to analyze."

        if not query:
            return "Please provide a question about the website content."

        try:
            # Process the first URL
            url = urls[0]
            result = self.pipeline.run({
                "fetcher": {"urls": [url]},
                "prompt_builder": {
                    "query": query,
                    "website_url": url
                }
            })

            return result["llm"]["replies"][0].content

        except Exception as e:
            logger.error(f"Error in API run: {str(e)}")
            return f"Error processing request: {str(e)}"
```

### Enhanced Version with Multiple Website Support

```python
# enhanced_pipeline_wrapper.py
from typing import AsyncGenerator, List, Dict, Any
from haystack import Pipeline
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIChatGenerator
from haystack.components.routers import DocumentSplitter
from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message
import re
import logging
from urllib.parse import urlparse
import asyncio

logger = logging.getLogger(__name__)

class EnhancedPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup enhanced chat-with-website pipeline"""
        # Initialize components
        self.fetcher = LinkContentFetcher()
        self.converter = HTMLToDocument()
        self.cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=True,
            keep_only_main_content=True
        )
        self.splitter = DocumentSplitter(split_by="word", split_length=1000, split_overlap=200)
        self.prompt_builder = PromptBuilder(
            template="""You are a helpful assistant that answers questions based on website content.

{{#if multiple_websites}}
You are analyzing content from multiple websites:
{{#each websites}}
- {{this.url}}: {{this.title}}
{{/each}}

Combined Content:
{{documents}}

{{else}}
Website: {{website_url}}
{{#if website_title}}
Title: {{website_title}}
{{/if}}

Content:
{{documents}}
{{/if}}

User Question: {{query}}

Please provide a comprehensive answer based on the website content(s). If comparing multiple websites, highlight key differences and similarities. If the content doesn't contain enough information to answer the question, please say so.

Answer:"""
        )
        self.llm = OpenAIChatGenerator(
            model="gpt-4",
            streaming_callback=lambda x: None
        )

        # Build pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("fetcher", self.fetcher)
        self.pipeline.add_component("converter", self.converter)
        self.pipeline.add_component("cleaner", self.cleaner)
        self.pipeline.add_component("splitter", self.splitter)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)

        # Connect components
        self.pipeline.connect("fetcher.content", "converter")
        self.pipeline.connect("converter.documents", "cleaner")
        self.pipeline.connect("cleaner.documents", "splitter")
        self.pipeline.connect("splitter.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from user message"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)

        # Validate URLs
        valid_urls = []
        for url in urls:
            try:
                parsed = urlparse(url)
                if all([parsed.scheme, parsed.netloc]):
                    valid_urls.append(url)
            except Exception:
                continue

        return valid_urls

    async def fetch_website_info(self, url: str) -> Dict[str, str]:
        """Fetch basic information about a website"""
        try:
            result = self.pipeline.run({
                "fetcher": {"urls": [url]},
                "converter": {},
                "cleaner": {}
            })

            documents = result["cleaner"]["documents"]
            if documents:
                # Extract title from first document
                content = documents[0].content
                title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE)
                title = title_match.group(1) if title_match else url

                return {"url": url, "title": title}
            else:
                return {"url": url, "title": url}

        except Exception as e:
            logger.error(f"Error fetching website info for {url}: {str(e)}")
            return {"url": url, "title": url}

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        """Run chat completion with enhanced website support"""
        question = get_last_user_message(messages)

        # Extract URLs from user message
        urls = self.extract_urls(question)

        if not urls:
            yield "Please provide a website URL to chat about. For example:\n" \
                  "- 'Tell me about https://example.com'\n" \
                  "- 'Compare https://site1.com and https://site2.com'"
            return

        try:
            if len(urls) == 1:
                # Single website mode
                url = urls[0]
                website_info = await self.fetch_website_info(url)

                yield f"📄 Analyzing {website_info['title']}...\n\n"

                result = async_streaming_generator(
                    pipeline=self.pipeline,
                    pipeline_run_args={
                        "fetcher": {"urls": [url]},
                        "prompt_builder": {
                            "query": question,
                            "website_url": url,
                            "website_title": website_info.get("title", ""),
                            "multiple_websites": False
                        }
                    },
                )

                async for chunk in result:
                    yield chunk

            else:
                # Multiple websites mode
                yield f"📚 Analyzing {len(urls)} websites...\n\n"

                # Fetch info for all websites
                website_infos = await asyncio.gather(
                    *[self.fetch_website_info(url) for url in urls]
                )

                # Fetch all websites
                result = async_streaming_generator(
                    pipeline=self.pipeline,
                    pipeline_run_args={
                        "fetcher": {"urls": urls},
                        "prompt_builder": {
                            "query": question,
                            "websites": website_infos,
                            "multiple_websites": True
                        }
                    },
                )

                async for chunk in result:
                    yield chunk

        except Exception as e:
            logger.error(f"Error processing websites: {str(e)}")
            yield f"Sorry, I encountered an error while processing the websites: {str(e)}"

    def run_api(self, query: str = "", urls: List[str] = None) -> Dict[str, Any]:
        """Run API endpoint for website chat"""
        if not urls:
            return {"error": "Please provide at least one URL to analyze."}

        if not query:
            return {"error": "Please provide a question about the website content."}

        try:
            if len(urls) == 1:
                # Single website
                url = urls[0]
                result = self.pipeline.run({
                    "fetcher": {"urls": [url]},
                    "prompt_builder": {
                        "query": query,
                        "website_url": url,
                        "multiple_websites": False
                    }
                })
            else:
                # Multiple websites
                website_infos = [self.fetch_website_info(url) for url in urls]
                result = self.pipeline.run({
                    "fetcher": {"urls": urls},
                    "prompt_builder": {
                        "query": query,
                        "websites": website_infos,
                        "multiple_websites": True
                    }
                })

            return {
                "answer": result["llm"]["replies"][0].content,
                "urls_analyzed": urls,
                "query": query
            }

        except Exception as e:
            logger.error(f"Error in API run: {str(e)}")
            return {"error": f"Error processing request: {str(e)}"}
```

## Deployment

### Using the CLI

```bash
# Deploy the pipeline
hayhooks pipeline deploy-files -n chat-with-website ./chat-with-website --description "Chat with website content"

# Deploy the enhanced version
hayhooks pipeline deploy-files -n chat-with-website-enhanced ./chat-with-website-enhanced --description "Enhanced chat with multiple websites"
```

### Using YAML Configuration

```yaml
# chat-with-website.yml
name: chat-with-website
description: "Chat with website content"
inputs:
  - name: query
    type: str
    description: "Question about the website"
  - name: urls
    type: List[str]
    description: "List of website URLs to analyze"
outputs:
  - name: answer
    type: str
    description: "Answer based on website content"
```

```bash
# Deploy from YAML
hayhooks pipeline deploy-yaml -n chat-with-website chat-with-website.yml
```

## Usage Examples

### OpenWebUI Integration

```python
# Use with OpenWebUI
curl -X POST http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chat-with-website",
    "messages": [
      {
        "role": "user",
        "content": "What products does https://www.apple.com offer?"
      }
    ]
  }'
```

### Direct API Usage

```bash
# Single website analysis
curl -X POST http://localhost:1416/chat-with-website/run \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is the main purpose of this website?",
    "urls": ["https://github.com"]
  }'

# Multiple websites comparison
curl -X POST http://localhost:1416/chat-with-website-enhanced/run \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Compare the features of these two cloud platforms",
    "urls": ["https://aws.amazon.com", "https://azure.microsoft.com"]
  }'
```

### CLI Usage

```bash
# Basic website chat
hayhooks pipeline run chat-with-website --param 'query="What services does this company offer?"' --param 'urls=["https://microsoft.com"]'

# Compare multiple websites
hayhooks pipeline run chat-with-website-enhanced --param 'query="Compare these two tech companies"' --param 'urls=["https://google.com", "https://apple.com"]'
```

## Testing

### Test Script

```python
# test_chat_with_website.py
import requests
import json

def test_single_website():
    """Test single website analysis"""
    url = "http://localhost:1416/chat-with-website/run"
    payload = {
        "query": "What is this website about?",
        "urls": ["https://python.org"]
    }

    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_multiple_websites():
    """Test multiple websites comparison"""
    url = "http://localhost:1416/chat-with-website-enhanced/run"
    payload = {
        "query": "Compare these two tech companies",
        "urls": ["https://google.com", "https://microsoft.com"]
    }

    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_openwebui_integration():
    """Test OpenWebUI integration"""
    url = "http://localhost:1416/v1/chat/completions"
    payload = {
        "model": "chat-with-website",
        "messages": [
            {
                "role": "user",
                "content": "Tell me about https://github.com"
            }
        ]
    }

    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("Testing single website...")
    test_single_website()

    print("\nTesting multiple websites...")
    test_multiple_websites()

    print("\nTesting OpenWebUI integration...")
    test_openwebui_integration()
```

## Troubleshooting

### Common Issues

1. **Website Not Accessible**
   - Check if the website is publicly accessible
   - Verify the URL format
   - Check for rate limiting or blocking

2. **Content Extraction Issues**
   - Some websites may have complex JavaScript-rendered content
   - Consider using a headless browser for complex sites
   - Check if the website has anti-scraping measures

3. **Performance Issues**
   - Large websites may take time to process
   - Consider implementing caching
   - Use content summarization for large pages

### Debugging

```python
# Add debug logging to your pipeline
import logging

logging.basicConfig(level=logging.DEBUG)

# In your pipeline wrapper, add logging
logger = logging.getLogger(__name__)

class PipelineWrapper(BasePipelineWrapper):
    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict):
        question = get_last_user_message(messages)
        urls = self.extract_urls(question)

        logger.debug(f"Extracted URLs: {urls}")

        for url in urls:
            logger.info(f"Processing URL: {url}")
            # ... rest of the code
```

## Extensions and Customizations

### Adding Content Summarization

```python
# Add a summarization component
from haystack.components.generators import OpenAIGenerator

class SummarizingPipelineWrapper(BasePipelineWrapper):
    def setup(self):
        # ... existing setup ...

        # Add summarizer
        self.summarizer = OpenAIGenerator(
            model="gpt-3.5-turbo",
            generation_kwargs={"max_tokens": 200}
        )

        # Add summarization prompt
        self.summary_prompt = PromptBuilder(
            template="Summarize this content in 2-3 sentences: {{documents}}"
        )

        # ... build pipeline with summarization ...

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict):
        # ... existing code ...

        # Add summarization for large content
        if len(content) > 5000:  # If content is too long
            summary = await self.summarize_content(content)
            content = summary

        # ... rest of the code ...
```

### Adding Caching

```python
# Add caching to improve performance
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

class CachedPipelineWrapper(BasePipelineWrapper):
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    @cache(expire=3600)  # Cache for 1 hour
    async def fetch_website_content(self, url: str):
        """Fetch website content with caching"""
        # ... existing fetching logic ...

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict):
        # Use cached content fetching
        for url in urls:
            content = await self.fetch_website_content(url)
            # ... rest of the code ...
```

## Best Practices

### 1. URL Validation
- Always validate URLs before processing
- Handle malformed URLs gracefully
- Check for accessibility before full processing

### 2. Content Processing
- Clean HTML content properly
- Handle different content types
- Respect website terms of service

### 3. Performance
- Implement caching for frequently accessed sites
- Use streaming for large content
- Set appropriate timeouts

### 4. Error Handling
- Provide clear error messages
- Log errors for debugging
- Implement retry logic for transient failures

## Next Steps

- [RAG System Example](rag-system.md) - RAG implementation
- [Async Operations](async-operations.md) - Asynchronous patterns
- [OpenWebUI Events](openwebui-events.md) - OpenWebUI integration