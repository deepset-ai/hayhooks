# OpenWebUI Events Example

This example demonstrates how to implement advanced OpenWebUI integration with Hayhooks, including event handling, tool call interception, and enhanced user experience features.

## Overview

OpenWebUI events allow you to:

- Send real-time status updates to users
- Intercept tool calls for better feedback
- Handle file uploads with progress tracking
- Provide custom error handling and recovery
- Create interactive conversational experiences

## Basic Event Integration

### Simple Event Pipeline

```python
# pipeline_wrapper.py
from typing import AsyncGenerator, List, Dict, Any
from haystack import Pipeline
from haystack.components import PromptBuilder, OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message, send_openwebui_event
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

class EventPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup pipeline with OpenWebUI event support"""
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
        """Run chat completion with OpenWebUI events"""
        question = get_last_user_message(messages)

        # Send loading start event
        await send_openwebui_event(
            event_type="loading_start",
            data={"message": "Processing your request..."}
        )

        try:
            # Send processing started event
            await send_openwebui_event(
                event_type="message_update",
                data={"message": "🔄 Processing your question..."}
            )

            # Add a small delay to show the loading state
            await asyncio.sleep(0.5)

            # Process the request
            result = async_streaming_generator(
                pipeline=self.pipeline,
                pipeline_run_args={"prompt_builder": {"query": question}},
            )

            # Send completion started event
            await send_openwebui_event(
                event_type="message_update",
                data={"message": "✅ Generating response..."}
            )

            # Stream the response
            response_text = ""
            async for chunk in result:
                response_text += chunk
                yield chunk

            # Send completion event
            await send_openwebui_event(
                event_type="loading_end",
                data={"message": "Request completed successfully"}
            )

        except Exception as e:
            # Send error event
            await send_openwebui_event(
                event_type="toast_notification",
                data={
                    "message": f"Error: {str(e)}",
                    "type": "error"
                }
            )
            raise

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> str:
        """Sync version for compatibility"""
        question = get_last_user_message(messages)

        result = self.pipeline.run({
            "prompt_builder": {"query": question}
        })

        return result["llm"]["replies"][0].content
```

### Enhanced Event Pipeline with Tool Calls

```python
# enhanced_event_pipeline.py
from typing import AsyncGenerator, List, Dict, Any, Optional
from haystack import Pipeline
from haystack.components import PromptBuilder, OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message, send_openwebui_event
from fastapi import UploadFile
import asyncio
import logging
import time
import json

logger = logging.getLogger(__name__)

class EnhancedEventPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup enhanced pipeline with tool call events"""
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

    def on_tool_call_start(self, tool_name: str, arguments: dict, tool_id: str):
        """Handle tool call start event"""
        asyncio.create_task(send_openwebui_event(
            event_type="tool_call_start",
            data={
                "tool_name": tool_name,
                "tool_id": tool_id,
                "arguments": arguments,
                "message": f"🔧 Using {tool_name}..."
            }
        ))

    def on_tool_call_end(self, tool_name: str, arguments: dict, result: dict, error: bool):
        """Handle tool call end event"""
        if error:
            asyncio.create_task(send_openwebui_event(
                event_type="tool_call_error",
                data={
                    "tool_name": tool_name,
                    "error": str(result),
                    "message": f"❌ {tool_name} failed"
                }
            ))
        else:
            asyncio.create_task(send_openwebui_event(
                event_type="tool_call_success",
                data={
                    "tool_name": tool_name,
                    "result": str(result)[:100] + "..." if len(str(result)) > 100 else str(result),
                    "message": f"✅ {tool_name} completed"
                }
            ))

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        """Run chat completion with enhanced events"""
        question = get_last_user_message(messages)

        # Extract files from body if present
        files = body.get("files", [])

        # Send initial events
        await send_openwebui_event(
            event_type="loading_start",
            data={"message": "🚀 Starting request..."}
        )

        try:
            # Process files if present
            if files:
                await send_openwebui_event(
                    event_type="message_update",
                    data={"message": f"📁 Processing {len(files)} files..."}
                )

                for i, file_info in enumerate(files):
                    await send_openwebui_event(
                        event_type="file_processing_start",
                        data={
                            "filename": file_info.get("filename", f"file_{i}"),
                            "file_number": i + 1,
                            "total_files": len(files)
                        }
                    )

                    # Simulate file processing
                    await asyncio.sleep(0.5)

                    await send_openwebui_event(
                        event_type="file_processing_complete",
                        data={
                            "filename": file_info.get("filename", f"file_{i}"),
                            "file_number": i + 1
                        }
                    )

                await send_openwebui_event(
                    event_type="message_update",
                    data={"message": "✅ All files processed"}
                )

            # Process the question
            await send_openwebui_event(
                event_type="message_update",
                data={"message": "🧠 Analyzing your question..."}
            )

            # Use streaming generator with tool call callbacks
            result = async_streaming_generator(
                pipeline=self.pipeline,
                pipeline_run_args={"prompt_builder": {"query": question}},
                on_tool_call_start=self.on_tool_call_start,
                on_tool_call_end=self.on_tool_call_end
            )

            await send_openwebui_event(
                event_type="message_update",
                data={"message": "✍️ Generating response..."}
            )

            # Stream the response
            response_text = ""
            async for chunk in result:
                response_text += chunk
                yield chunk

            # Send completion events
            await send_openwebui_event(
                event_type="loading_end",
                data={"message": "✅ Request completed successfully"}
            )

            # Send summary event
            await send_openwebui_event(
                event_type="session_summary",
                data={
                    "question_length": len(question),
                    "response_length": len(response_text),
                    "files_processed": len(files),
                    "processing_time": time.time()
                }
            )

        except Exception as e:
            logger.error(f"Error in enhanced event pipeline: {str(e)}")

            # Send error events
            await send_openwebui_event(
                event_type="loading_end",
                data={"message": "❌ Request failed"}
            )

            await send_openwebui_event(
                event_type="toast_notification",
                data={
                    "message": f"Error: {str(e)}",
                    "type": "error",
                    "duration": 5000
                }
            )

            # Send recovery suggestion
            await send_openwebui_event(
                event_type="suggestion",
                data={
                    "message": "Would you like me to try again with a simpler approach?",
                    "action": "retry_simple"
                }
            )

            raise
```

## Interactive Features Pipeline

### Pipeline with Interactive Elements

```python
# interactive_pipeline.py
from typing import AsyncGenerator, List, Dict, Any, Optional
from haystack import Pipeline
from haystack.components import PromptBuilder, OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message, send_openwebui_event
import asyncio
import logging
import time
import json

logger = logging.getLogger(__name__)

class InteractivePipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup interactive pipeline with rich event support"""
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

    async def send_progress_event(self, current: int, total: int, message: str):
        """Send progress update event"""
        await send_openwebui_event(
            event_type="progress_update",
            data={
                "current": current,
                "total": total,
                "percentage": int((current / total) * 100),
                "message": message
            }
        )

    async def send_interactive_suggestion(self, question: str, context: str = ""):
        """Send interactive suggestion based on context"""
        suggestions = self._generate_suggestions(question, context)

        await send_openwebui_event(
            event_type="interactive_suggestions",
            data={
                "suggestions": suggestions,
                "context": "follow-up"
            }
        )

    def _generate_suggestions(self, question: str, context: str = "") -> List[str]:
        """Generate follow-up suggestions based on question"""
        question_lower = question.lower()

        if "what" in question_lower:
            return [
                "Can you explain that in more detail?",
                "What are the key benefits?",
                "How does this compare to alternatives?"
            ]
        elif "how" in question_lower:
            return [
                "Can you show me an example?",
                "What are the steps involved?",
                "Are there any common pitfalls?"
            ]
        elif "why" in question_lower:
            return [
                "What are the main reasons?",
                "Are there any disadvantages?",
                "What's the historical context?"
            ]
        else:
            return [
                "Can you elaborate on that?",
                "What are the implications?",
                "Do you have any practical examples?"
            ]

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        """Run chat completion with interactive features"""
        question = get_last_user_message(messages)

        # Get conversation context
        context_messages = messages[:-1]  # All messages except the last one
        context = " ".join([msg.get("content", "") for msg in context_messages[-3:]])

        # Send welcome event with personality
        await send_openwebui_event(
            event_type="personalized_greeting",
            data={
                "message": "👋 I'm here to help!",
                "mood": "helpful",
                "context": "general"
            }
        )

        try:
            # Simulate processing steps with progress
            steps = [
                "🔍 Understanding your question",
                "📚 Gathering relevant information",
                "🧠 Processing and analyzing",
                "✍️ Crafting response"
            ]

            for i, step in enumerate(steps):
                await self.send_progress_event(
                    current=i + 1,
                    total=len(steps),
                    message=step
                )
                await asyncio.sleep(0.3)  # Simulate processing time

            # Send analysis summary
            await send_openwebui_event(
                event_type="analysis_summary",
                data={
                    "question_type": self._analyze_question_type(question),
                    "complexity": "medium",
                    "estimated_time": "2-3 seconds",
                    "confidence": "high"
                }
            )

            # Process the request
            result = async_streaming_generator(
                pipeline=self.pipeline,
                pipeline_run_args={"prompt_builder": {"query": question}},
            )

            # Stream the response
            response_text = ""
            word_count = 0
            async for chunk in result:
                response_text += chunk
                word_count += len(chunk.split())

                # Send word count updates periodically
                if word_count % 20 == 0:
                    await send_openwebui_event(
                        event_type="writing_progress",
                        data={
                            "word_count": word_count,
                            "message": f"✍️ Writing... ({word_count} words)"
                        }
                    )

                yield chunk

            # Send completion events
            await send_openwebui_event(
                event_type="completion_summary",
                data={
                    "total_words": word_count,
                    "processing_time": "2.5s",
                    "confidence": "high",
                    "satisfaction_check": True
                }
            )

            # Send interactive suggestions
            await self.send_interactive_suggestion(question, context)

            # Send helpful resources
            await send_openwebui_event(
                event_type="helpful_resources",
                data={
                    "resources": [
                        {
                            "title": "Learn more about this topic",
                            "url": "#",
                            "type": "documentation"
                        },
                        {
                            "title": "Related examples",
                            "url": "#",
                            "type": "examples"
                        }
                    ]
                }
            )

        except Exception as e:
            logger.error(f"Error in interactive pipeline: {str(e)}")

            # Send error with personality
            await send_openwebui_event(
                event_type="friendly_error",
                data={
                    "message": "😅 Oops! I ran into an issue",
                    "error": str(e),
                    "suggestion": "Would you like me to try a different approach?",
                    "confidence": "still_high"
                }
            )

            # Send recovery options
            await send_openwebui_event(
                event_type="recovery_options",
                data={
                    "options": [
                        {"text": "Try again", "action": "retry"},
                        {"text": "Simplify question", "action": "simplify"},
                        {"text": "Get help", "action": "help"}
                    ]
                }
            )

            raise

    def _analyze_question_type(self, question: str) -> str:
        """Analyze the type of question"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["what", "who", "when", "where", "why", "how"]):
            return "informational"
        elif any(word in question_lower for word in ["can", "could", "would", "should"]):
            return "advisory"
        elif any(word in question_lower for word in ["help", "assist", "support"]):
            return "help"
        elif "?" in question:
            return "question"
        else:
            return "statement"
```

## File Upload with Progress

### Enhanced File Processing Pipeline

```python
# file_upload_pipeline.py
from typing import AsyncGenerator, List, Dict, Any, Optional
from fastapi import UploadFile
from haystack import Pipeline
from haystack.components import PromptBuilder, OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message, send_openwebui_event
import asyncio
import logging
import time
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class FileUploadPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup pipeline with enhanced file upload events"""
        # Initialize components
        self.prompt_builder = PromptBuilder(
            template="You are a helpful assistant. Answer: {{query}}\n\nFiles: {{file_summary}}"
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

    async def process_file_with_events(self, file: UploadFile, file_index: int, total_files: int) -> Dict[str, Any]:
        """Process a single file with event updates"""
        try:
            # Send file processing start
            await send_openwebui_event(
                event_type="file_processing_start",
                data={
                    "filename": file.filename,
                    "file_size": file.size,
                    "file_index": file_index,
                    "total_files": total_files,
                    "estimated_time": "2-3 seconds"
                }
            )

            # Simulate processing progress
            progress_steps = [
                "📁 Reading file...",
                "🔍 Analyzing content...",
                "📊 Extracting information...",
                "✅ Processing complete"
            ]

            for i, step in enumerate(progress_steps):
                await send_openwebui_event(
                    event_type="file_progress",
                    data={
                        "filename": file.filename,
                        "step": i + 1,
                        "total_steps": len(progress_steps),
                        "message": step,
                        "progress": ((i + 1) / len(progress_steps)) * 100
                    }
                )
                await asyncio.sleep(0.2)

            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            # Extract file information
            file_info = {
                "filename": file.filename,
                "size": len(content),
                "type": Path(file.filename).suffix.lower(),
                "content_preview": content[:200].decode('utf-8', errors='ignore') + "..." if len(content) > 200 else content.decode('utf-8', errors='ignore')
            }

            # Clean up
            os.unlink(tmp_file_path)

            # Send file processing complete
            await send_openwebui_event(
                event_type="file_processing_complete",
                data={
                    "filename": file.filename,
                    "file_info": file_info,
                    "processing_time": "1.5s",
                    "status": "success"
                }
            )

            return file_info

        except Exception as e:
            # Send file processing error
            await send_openwebui_event(
                event_type="file_processing_error",
                data={
                    "filename": file.filename,
                    "error": str(e),
                    "suggestion": "Please try uploading the file again"
                }
            )
            raise

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        """Run chat completion with enhanced file upload events"""
        question = get_last_user_message(messages)

        # Extract files from body
        files = body.get("files", [])

        try:
            if files:
                # Send file processing start
                await send_openwebui_event(
                    event_type="batch_file_start",
                    data={
                        "total_files": len(files),
                        "total_size": sum(f.get("size", 0) for f in files),
                        "estimated_time": f"{len(files) * 2} seconds"
                    }
                )

                # Process files
                file_infos = []
                for i, file_data in enumerate(files):
                    # Create UploadFile from data
                    file = UploadFile(
                        filename=file_data.get("filename", f"file_{i}"),
                        file=await self._get_file_content(file_data),
                        size=file_data.get("size", 0)
                    )

                    file_info = await self.process_file_with_events(file, i + 1, len(files))
                    file_infos.append(file_info)

                # Send batch processing complete
                await send_openwebui_event(
                    event_type="batch_file_complete",
                    data={
                        "processed_files": len(file_infos),
                        "total_size": sum(f["size"] for f in file_infos),
                        "file_types": list(set(f["type"] for f in file_infos)),
                        "processing_time": f"{len(files) * 1.5}s"
                    }
                )

                # Create file summary
                file_summary = "\n".join([
                    f"- {info['filename']} ({info['size']} bytes, {info['type']})"
                    for info in file_infos
                ])

            else:
                file_summary = "No files uploaded"

            # Send question processing start
            await send_openwebui_event(
                event_type="question_processing_start",
                data={
                    "question": question[:50] + "..." if len(question) > 50 else question,
                    "has_files": len(files) > 0,
                    "estimated_time": "1-2 seconds"
                }
            )

            # Process the question
            result = async_streaming_generator(
                pipeline=self.pipeline,
                pipeline_run_args={
                    "prompt_builder": {
                        "query": question,
                        "file_summary": file_summary
                    }
                },
            )

            # Send response generation start
            await send_openwebui_event(
                event_type="response_generation_start",
                data={
                    "message": "✍️ Generating response based on your files and question..."
                }
            )

            # Stream the response
            response_text = ""
            async for chunk in result:
                response_text += chunk
                yield chunk

            # Send completion events
            await send_openwebui_event(
                event_type="session_complete",
                data={
                    "files_processed": len(files),
                    "response_length": len(response_text),
                    "total_time": "3-4 seconds",
                    "satisfaction": "high"
                }
            )

            # Send follow-up suggestions
            await send_openwebui_event(
                event_type="follow_up_suggestions",
                data={
                    "suggestions": [
                        "Ask another question about these files",
                        "Upload more files for comparison",
                        "Get a summary of the uploaded content"
                    ]
                }
            )

        except Exception as e:
            logger.error(f"Error in file upload pipeline: {str(e)}")

            # Send error with recovery options
            await send_openwebui_event(
                event_type="file_upload_error",
                data={
                    "error": str(e),
                    "recovery_options": [
                        "Try uploading files again",
                        "Continue without files",
                        "Get help with file uploads"
                    ]
                }
            )

            raise

    async def _get_file_content(self, file_data: Dict[str, Any]):
        """Get file content from file data"""
        import io

        content = file_data.get("content", "")
        if isinstance(content, str):
            # Handle base64 content
            import base64
            try:
                decoded_content = base64.b64decode(content)
                return io.BytesIO(decoded_content)
            except:
                return io.BytesIO(content.encode())
        else:
            return io.BytesIO(content)
```

## Testing and Usage

### OpenWebUI Integration Test

```python
# test_openwebui_events.py
import asyncio
import aiohttp
import json

async def test_basic_events():
    """Test basic event integration"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "event-pipeline",
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me about artificial intelligence"
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

async def test_enhanced_events():
    """Test enhanced events with tool calls"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "enhanced-event-pipeline",
            "messages": [
                {
                    "role": "user",
                    "content": "What can you tell me about machine learning?"
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

async def test_file_upload_events():
    """Test file upload with events"""
    async with aiohttp.ClientSession() as session:
        # Simulate file upload
        payload = {
            "model": "file-upload-pipeline",
            "messages": [
                {
                    "role": "user",
                    "content": "What's in these files?"
                }
            ],
            "files": [
                {
                    "filename": "document.txt",
                    "content": "VGhpcyBpcyBhIHRlc3QgZG9jdW1lbnQgd2l0aCBzb21lIHNhbXBsZSB0ZXh0Lg==",
                    "size": 50,
                    "type": ".txt"
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

if __name__ == "__main__":
    asyncio.run(test_basic_events())
    print("\n" + "="*50 + "\n")
    asyncio.run(test_enhanced_events())
    print("\n" + "="*50 + "\n")
    asyncio.run(test_file_upload_events())
```

## Best Practices

### 1. Event Design

- Use descriptive event names that indicate their purpose
- Include relevant data in each event for better UX
- Keep events lightweight and focused
- Use consistent event naming conventions

### 2. User Experience

- Provide clear feedback for all operations
- Use appropriate timing for events (not too fast, not too slow)
- Include progress indicators for long-running operations
- Offer recovery options for errors

### 3. Performance

- Use async/await for non-blocking event sending
- Batch events when possible to reduce overhead
- Monitor event frequency and impact on performance
- Use appropriate timeouts for event operations

### 4. Error Handling

- Handle event sending errors gracefully
- Provide fallback behavior when events fail
- Log event failures for debugging
- Don't let event failures break the main functionality

### 5. Testing

- Test events with different OpenWebUI versions
- Verify event data structure and content
- Test error scenarios and recovery
- Monitor event delivery and timing

## Event Types Reference

### Standard Events

- `loading_start` - Show loading spinner
- `loading_end` - Hide loading spinner
- `message_update` - Update status message
- `toast_notification` - Show toast notification
- `progress_update` - Update progress bar
- `error` - Show error message

### Custom Events

- `tool_call_start` - Tool call started
- `tool_call_end` - Tool call completed
- `file_processing_start` - File processing started
- `file_processing_complete` - File processing completed
- `session_summary` - Session summary
- `interactive_suggestions` - Follow-up suggestions

## Next Steps

- [Chat with Website Example](chat-with-website.md) - Website analysis
- [RAG System Example](rag-system.md) - Document Q&A
- [Async Operations](async-operations.md) - Asynchronous patterns
