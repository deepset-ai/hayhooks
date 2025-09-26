# Async Operations Example

This example demonstrates how to implement asynchronous operations in Hayhooks pipelines, including background processing, streaming responses, and concurrent execution.

## Overview

Async operations allow you to:

- Process long-running tasks without blocking
- Stream responses in real-time
- Handle multiple requests concurrently
- Implement background job processing
- Improve overall system performance

## Basic Async Pipeline

### Simple Async Chat Pipeline

```python
# pipeline_wrapper.py
from typing import AsyncGenerator, List, Dict, Any
from haystack import Pipeline
from haystack.components import PromptBuilder, OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message
import asyncio
import logging

logger = logging.getLogger(__name__)

class AsyncPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup basic async pipeline"""
        # Initialize components
        self.prompt_builder = PromptBuilder(
            template="You are a helpful assistant. Answer: {{query}}"
        )
        self.llm = OpenAIChatGenerator(
            model="gpt-4",
            streaming_callback=lambda x: None
        )

        # Build pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        self.pipeline.connect("prompt_builder", "llm")

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        """Run async chat completion with streaming"""
        question = get_last_user_message(messages)

        logger.info(f"Processing question: {question}")

        # Use async streaming generator
        result = async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"prompt_builder": {"query": question}},
        )

        # Stream the response
        async for chunk in result:
            yield chunk

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> str:
        """Sync version for compatibility"""
        question = get_last_user_message(messages)

        result = self.pipeline.run({
            "prompt_builder": {"query": question}
        })

        return result["llm"]["replies"][0].content
```

### Enhanced Async Pipeline with Background Tasks

```python
# enhanced_async_pipeline.py
from typing import AsyncGenerator, List, Dict, Any, Optional
from haystack import Pipeline
from haystack.components import PromptBuilder, OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message
from fastapi import UploadFile
import asyncio
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class EnhancedAsyncPipelineWrapper(BasePipelineWrapper):
    def __init__(self):
        self.background_tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.task_status = {}

    def setup(self) -> None:
        """Setup enhanced async pipeline with background tasks"""
        # Initialize components
        self.prompt_builder = PromptBuilder(
            template="You are a helpful assistant. Answer: {{query}}"
        )
        self.llm = OpenAIChatGenerator(
            model="gpt-4",
            streaming_callback=lambda x: None
        )

        # Build pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        self.pipeline.connect("prompt_builder", "llm")

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        """Run async chat completion with task tracking"""
        task_id = str(uuid.uuid4())
        self.task_status[task_id] = {
            "status": "started",
            "start_time": time.time(),
            "messages": len(messages)
        }

        try:
            question = get_last_user_message(messages)

            self.task_status[task_id]["status"] = "processing"
            logger.info(f"Task {task_id}: Processing question: {question}")

            # Use async streaming generator
            result = async_streaming_generator(
                pipeline=self.pipeline,
                pipeline_run_args={"prompt_builder": {"query": question}},
            )

            # Stream the response
            async for chunk in result:
                yield chunk

            self.task_status[task_id]["status"] = "completed"
            self.task_status[task_id]["end_time"] = time.time()

        except Exception as e:
            self.task_status[task_id]["status"] = "error"
            self.task_status[task_id]["error"] = str(e)
            logger.error(f"Task {task_id} failed: {str(e)}")
            raise

    async def process_files_async(self, files: List[UploadFile]) -> AsyncGenerator:
        """Process files asynchronously"""
        task_id = str(uuid.uuid4())
        self.task_status[task_id] = {
            "status": "started",
            "start_time": time.time(),
            "files": len(files)
        }

        try:
            yield f"Processing {len(files)} files...\n"

            # Process files in parallel
            async def process_single_file(file: UploadFile):
                try:
                    content = await file.read()
                    # Simulate processing
                    await asyncio.sleep(1)
                    return {
                        "filename": file.filename,
                        "size": len(content),
                        "status": "processed"
                    }
                except Exception as e:
                    return {
                        "filename": file.filename,
                        "status": "error",
                        "error": str(e)
                    }

            # Create tasks for all files
            tasks = [process_single_file(file) for file in files]
            results = await asyncio.gather(*tasks)

            # Process results
            processed_count = sum(1 for r in results if r["status"] == "processed")
            error_count = len(results) - processed_count

            yield f"\nProcessed {processed_count} files successfully"
            if error_count > 0:
                yield f"\n{error_count} files had errors"

            # Yield individual results
            for result in results:
                if result["status"] == "processed":
                    yield f"\n✓ {result['filename']} ({result['size']} bytes)"
                else:
                    yield f"\n✗ {result['filename']}: {result['error']}"

            self.task_status[task_id]["status"] = "completed"
            self.task_status[task_id]["end_time"] = time.time()
            self.task_status[task_id]["results"] = results

        except Exception as e:
            self.task_status[task_id]["status"] = "error"
            self.task_status[task_id]["error"] = str(e)
            logger.error(f"File processing task {task_id} failed: {str(e)}")
            yield f"\nError: {str(e)}"

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a background task"""
        return self.task_status.get(task_id, {"error": "Task not found"})

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tasks"""
        return self.task_status

    def cleanup_old_tasks(self, max_age: int = 3600):
        """Clean up tasks older than max_age seconds"""
        current_time = time.time()
        old_tasks = [
            task_id for task_id, task_data in self.task_status.items()
            if current_time - task_data.get("end_time", current_time) > max_age
        ]

        for task_id in old_tasks:
            del self.task_status[task_id]

        logger.info(f"Cleaned up {len(old_tasks)} old tasks")
```

## Batch Processing Pipeline

### Concurrent Batch Processing

```python
# batch_async_pipeline.py
from typing import AsyncGenerator, List, Dict, Any, Optional
from haystack import Pipeline
from haystack.components import PromptBuilder, OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class BatchAsyncPipelineWrapper(BasePipelineWrapper):
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.batch_queue = asyncio.Queue()
        self.processing = False

    def setup(self) -> None:
        """Setup batch processing pipeline"""
        # Initialize components
        self.prompt_builder = PromptBuilder(
            template="Process this item: {{item}}\n\nInstruction: {{instruction}}"
        )
        self.llm = OpenAIChatGenerator(
            model="gpt-3.5-turbo",
            streaming_callback=lambda x: None
        )

        # Build pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        self.pipeline.connect("prompt_builder", "llm")

    async def process_batch_async(self, items: List[str], instruction: str) -> AsyncGenerator:
        """Process a batch of items concurrently"""
        batch_id = f"batch_{int(time.time())}"
        total_items = len(items)

        yield f"Starting batch processing of {total_items} items...\n"

        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent operations

        async def process_item(item: str, index: int) -> Dict[str, Any]:
            """Process a single item with semaphore"""
            async with semaphore:
                try:
                    yield f"\nProcessing item {index + 1}/{total_items}: {item[:50]}..."

                    # Process the item
                    result = await asyncio.to_thread(
                        self.pipeline.run,
                        {
                            "prompt_builder": {
                                "item": item,
                                "instruction": instruction
                            }
                        }
                    )

                    processed_result = result["llm"]["replies"][0]

                    return {
                        "item": item,
                        "result": processed_result,
                        "status": "success",
                        "index": index
                    }

                except Exception as e:
                    logger.error(f"Error processing item {index}: {str(e)}")
                    return {
                        "item": item,
                        "status": "error",
                        "error": str(e),
                        "index": index
                    }

        # Create tasks for all items
        tasks = [process_item(item, i) for i, item in enumerate(items)]

        # Process items and collect results
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                # Note: This is a simplified approach - in practice, you'd need
                # to modify the process_item function to return the result directly
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")

        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful

        yield f"\n\nBatch processing completed:"
        yield f"\n✓ Successfully processed: {successful} items"
        yield f"\n✗ Failed: {failed} items"

        if failed > 0:
            yield f"\n\nFailed items:"
            for result in results:
                if result["status"] == "error":
                    yield f"\n- {result['item']}: {result['error']}"

    async def process_batch_with_progress(self, items: List[str], instruction: str) -> AsyncGenerator:
        """Process batch with real-time progress updates"""
        total_items = len(items)
        completed = 0
        successful = 0
        failed = 0

        yield f"Starting batch processing of {total_items} items...\n"

        async def process_single_item_with_progress(item: str, index: int):
            nonlocal completed, successful, failed

            try:
                yield f"\n[{completed + 1}/{total_items}] Processing: {item[:50]}..."

                result = await asyncio.to_thread(
                    self.pipeline.run,
                    {
                        "prompt_builder": {
                            "item": item,
                            "instruction": instruction
                        }
                    }
                )

                completed += 1
                successful += 1

                yield f"\n✓ Completed item {index + 1}: {item[:30]}..."

                return result["llm"]["replies"][0]

            except Exception as e:
                completed += 1
                failed += 1
                yield f"\n✗ Failed item {index + 1}: {str(e)}"
                return None

        # Process items sequentially for better progress tracking
        results = []
        for i, item in enumerate(items):
            async for chunk in process_single_item_with_progress(item, i):
                yield chunk

        # Final summary
        yield f"\n\nBatch processing completed!"
        yield f"\nTotal items: {total_items}"
        yield f"\nSuccessful: {successful}"
        yield f"\nFailed: {failed}"
        yield f"\nSuccess rate: {(successful/total_items)*100:.1f}%"

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        """Run chat completion with batch processing support"""
        from hayhooks import get_last_user_message

        question = get_last_user_message(messages)

        # Check if this is a batch processing request
        if "batch_items" in body:
            items = body["batch_items"]
            instruction = body.get("instruction", "Process this item")

            async for chunk in self.process_batch_async(items, instruction):
                yield chunk
        else:
            # Regular single-item processing
            result = async_streaming_generator(
                pipeline=self.pipeline,
                pipeline_run_args={"prompt_builder": {"item": question, "instruction": "Answer this question"}},
            )

            async for chunk in result:
                yield chunk
```

## Background Job Processing

### Job Queue System

```python
# job_queue_pipeline.py
from typing import AsyncGenerator, List, Dict, Any, Optional
from haystack import Pipeline
from haystack.components import PromptBuilder, OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator
import asyncio
import time
import uuid
import json
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    id: str
    type: str
    data: Dict[str, Any]
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None

class JobQueuePipelineWrapper(BasePipelineWrapper):
    def __init__(self):
        self.jobs = {}
        self.job_queue = asyncio.Queue()
        self.worker_task = None
        self.max_retries = 3

    def setup(self) -> None:
        """Setup job queue pipeline"""
        # Initialize components
        self.prompt_builder = PromptBuilder(
            template="Process this job: {{data}}\n\nJob Type: {{job_type}}"
        )
        self.llm = OpenAIChatGenerator(
            model="gpt-4",
            streaming_callback=lambda x: None
        )

        # Build pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        self.pipeline.connect("prompt_builder", "llm")

        # Start the worker
        self.worker_task = asyncio.create_task(self.job_worker())

    async def job_worker(self):
        """Background job worker"""
        logger.info("Job worker started")

        while True:
            try:
                # Get job from queue
                job = await self.job_queue.get()

                # Process job
                await self.process_job(job)

                # Mark task as done
                self.job_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Job worker cancelled")
                break
            except Exception as e:
                logger.error(f"Job worker error: {str(e)}")
                await asyncio.sleep(1)  # Prevent rapid error loops

    async def process_job(self, job: Job):
        """Process a single job"""
        job.status = JobStatus.RUNNING
        job.started_at = time.time()

        logger.info(f"Processing job {job.id} of type {job.type}")

        try:
            if job.type == "text_processing":
                result = await self.process_text_job(job.data)
            elif job.type == "analysis":
                result = await self.process_analysis_job(job.data)
            elif job.type == "generation":
                result = await self.process_generation_job(job.data)
            else:
                raise ValueError(f"Unknown job type: {job.type}")

            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()

            logger.info(f"Job {job.id} completed successfully")

        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
            job.completed_at = time.time()

            logger.error(f"Job {job.id} failed: {str(e)}")

            # Retry logic
            if job.data.get("retry_count", 0) < self.max_retries:
                job.data["retry_count"] = job.data.get("retry_count", 0) + 1
                job.status = JobStatus.PENDING
                await self.job_queue.put(job)
                logger.info(f"Retrying job {job.id} (attempt {job.data['retry_count']})")

    async def process_text_job(self, data: Dict[str, Any]) -> str:
        """Process text processing job"""
        text = data.get("text", "")
        instruction = data.get("instruction", "Process this text")

        result = self.pipeline.run({
            "prompt_builder": {
                "data": text,
                "job_type": "text_processing"
            }
        })

        return result["llm"]["replies"][0]

    async def process_analysis_job(self, data: Dict[str, Any]) -> str:
        """Process analysis job"""
        content = data.get("content", "")
        analysis_type = data.get("analysis_type", "general")

        instruction = f"Analyze this content for {analysis_type}"

        result = self.pipeline.run({
            "prompt_builder": {
                "data": content,
                "job_type": "analysis"
            }
        })

        return result["llm"]["replies"][0]

    async def process_generation_job(self, data: Dict[str, Any]) -> str:
        """Process generation job"""
        prompt = data.get("prompt", "")
        parameters = data.get("parameters", {})

        result = self.pipeline.run({
            "prompt_builder": {
                "data": json.dumps({"prompt": prompt, "parameters": parameters}),
                "job_type": "generation"
            }
        })

        return result["llm"]["replies"][0]

    async def submit_job(self, job_type: str, data: Dict[str, Any]) -> str:
        """Submit a new job"""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            type=job_type,
            data=data,
            status=JobStatus.PENDING,
            created_at=time.time()
        )

        self.jobs[job_id] = job
        await self.job_queue.put(job)

        logger.info(f"Submitted job {job_id} of type {job_type}")
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        job = self.jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}

        return {
            "id": job.id,
            "type": job.type,
            "status": job.status.value,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "result": job.result,
            "error": job.error
        }

    async def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for job completion"""
        start_time = time.time()

        while True:
            job = self.jobs.get(job_id)
            if not job:
                return {"error": "Job not found"}

            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                return self.get_job_status(job_id)

            if timeout and (time.time() - start_time) > timeout:
                return {"error": "Timeout waiting for job completion"}

            await asyncio.sleep(0.1)

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        """Run chat completion with job queue support"""
        from hayhooks import get_last_user_message

        question = get_last_user_message(messages)

        # Check if this is a job submission
        if body.get("submit_job"):
            job_type = body.get("job_type", "text_processing")
            job_data = {"text": question, "instruction": body.get("instruction", "Process this text")}

            job_id = await self.submit_job(job_type, job_data)

            yield f"Job submitted with ID: {job_id}"
            yield f"Use the job status endpoint to check progress"

        elif body.get("job_id"):
            # Check job status
            job_id = body.get("job_id")
            status = self.get_job_status(job_id)

            if status.get("status") == "completed":
                yield f"Job completed: {status.get('result', '')}"
            elif status.get("status") == "failed":
                yield f"Job failed: {status.get('error', 'Unknown error')}"
            else:
                yield f"Job status: {status.get('status', 'unknown')}"

        else:
            # Regular processing
            result = async_streaming_generator(
                pipeline=self.pipeline,
                pipeline_run_args={"prompt_builder": {"data": question, "job_type": "chat"}},
            )

            async for chunk in result:
                yield chunk
```

## Testing and Usage

### Test Script

```python
# test_async_operations.py
import asyncio
import aiohttp
import json

async def test_basic_async():
    """Test basic async pipeline"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "async-pipeline",
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a joke about programming"
                }
            ]
        }

        async with session.post(
            "http://localhost:1416/v1/chat/completions",
            json=payload
        ) as response:
            print(f"Status: {response.status}")
            async for chunk in response.content:
                print(chunk.decode(), end="")

async def test_batch_processing():
    """Test batch processing"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "batch-async-pipeline",
            "messages": [
                {
                    "role": "user",
                    "content": "Process these items"
                }
            ],
            "batch_items": [
                "What is Python?",
                "What is machine learning?",
                "What is artificial intelligence?"
            ],
            "instruction": "Provide a brief definition"
        }

        async with session.post(
            "http://localhost:1416/v1/chat/completions",
            json=payload
        ) as response:
            print(f"Status: {response.status}")
            async for chunk in response.content:
                print(chunk.decode(), end="")

async def test_job_queue():
    """Test job queue system"""
    async with aiohttp.ClientSession() as session:
        # Submit a job
        payload = {
            "model": "job-queue-pipeline",
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze this text"
                }
            ],
            "submit_job": True,
            "job_type": "analysis",
            "instruction": "Extract key themes"
        }

        async with session.post(
            "http://localhost:1416/v1/chat/completions",
            json=payload
        ) as response:
            response_data = await response.json()
            print(f"Job submitted: {response_data}")

            # Check job status (you'd need to implement this endpoint)
            job_id = response_data.get("job_id")
            if job_id:
                status_url = f"http://localhost:1416/job-queue-pipeline/job/{job_id}"
                async with session.get(status_url) as status_response:
                    status_data = await status_response.json()
                    print(f"Job status: {status_data}")

if __name__ == "__main__":
    asyncio.run(test_basic_async())
    print("\n" + "="*50 + "\n")
    asyncio.run(test_batch_processing())
    print("\n" + "="*50 + "\n")
    asyncio.run(test_job_queue())
```

## Best Practices

### 1. Async Design Patterns

- Use async generators for streaming responses
- Implement proper error handling and cancellation
- Use semaphores to limit concurrent operations
- Monitor resource usage and performance

### 2. Background Processing

- Design idempotent operations for retries
- Implement proper job persistence and recovery
- Use appropriate timeouts and monitoring
- Consider using dedicated task queues for production

### 3. Performance Optimization

- Choose appropriate concurrency levels
- Use connection pooling for external services
- Implement caching for frequently accessed data
- Monitor memory usage and garbage collection

### 4. Error Handling

- Provide clear error messages for users
- Implement comprehensive logging
- Use circuit breakers for external services
- Design graceful degradation strategies

### 5. Testing

- Test both sync and async paths
- Test error conditions and edge cases
- Test under load and concurrency
- Monitor performance characteristics

## Next Steps

- [Chat with Website Example](chat-with-website.md) - Website analysis
- [RAG System Example](rag-system.md) - Document Q&A
- [OpenWebUI Events](openwebui-events.md) - OpenWebUI integration
