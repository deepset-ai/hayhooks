# Agent Deployment

Haystack Agents can be deployed using the same `PipelineWrapper` approach as regular pipelines. This section covers how to deploy Haystack agents with full functionality.

## Overview

Haystack Agents are intelligent systems that can use tools to accomplish tasks. With Hayhooks, you can deploy agents that:

- Use multiple tools to solve complex problems
- Provide conversational interfaces
- Support streaming responses
- Integrate with OpenAI-compatible chat systems

## Basic Agent Deployment

### Minimal Agent Example

```python
from typing import AsyncGenerator
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt="You're a helpful assistant that can answer questions and help with tasks.",
        )

    async def run_chat_completion_async(
        self, model: str, messages: list[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        chat_messages = [
            ChatMessage.from_openai_dict_format(message) for message in messages
        ]

        return async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={
                "messages": chat_messages,
            },
        )
```

## Agent with Tools

### Adding Tools to Your Agent

```python
from haystack.components.agents import Agent
from haystack.components.tools import OpenAPITool
from haystack.components.generators.chat import OpenAIChatGenerator

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Create tools
        weather_tool = OpenAPITool(
            name="weather",
            description="Get weather information for a location",
            openapi_spec={
                "openapi": "3.0.0",
                "info": {"title": "Weather API", "version": "1.0.0"},
                "paths": {
                    "/weather": {
                        "get": {
                            "parameters": [
                                {"name": "location", "in": "query", "required": True}
                            ]
                        }
                    }
                }
            }
        )

        # Create agent with tools
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o"),
            tools=[weather_tool],
            system_prompt="You're a helpful assistant. Use tools when needed to provide accurate information.",
        )
```

### Web Search Agent

```python
from haystack.components.agents import Agent
from haystack.components.tools import SerpDevTool
from haystack.components.generators.chat import OpenAIChatGenerator

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Web search tool
        search_tool = SerpDevTool()

        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o"),
            tools=[search_tool],
            system_prompt="You're a helpful assistant that can search the web for current information.",
        )

    async def run_chat_completion_async(
        self, model: str, messages: list[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        chat_messages = [
            ChatMessage.from_openai_dict_format(message) for message in messages
        ]

        return async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={
                "messages": chat_messages,
            },
        )
```

## Agent with Multiple Interfaces

### Full-Featured Agent

```python
from typing import AsyncGenerator, List, Optional, Dict, Any
from haystack.components.agents import Agent
from haystack.components.tools import SerpDevTool, OpenAPITool
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from hayhooks import BasePipelineWrapper, async_streaming_generator, log

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Create tools
        search_tool = SerpDevTool()
        calculator_tool = OpenAPITool(
            name="calculator",
            description="Perform mathematical calculations",
            openapi_spec={
                "openapi": "3.0.0",
                "info": {"title": "Calculator API", "version": "1.0.0"},
                "paths": {
                    "/calculate": {
                        "post": {
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "expression": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        )

        # Create agent
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o"),
            tools=[search_tool, calculator_tool],
            system_prompt="You're a helpful assistant that can search the web and perform calculations. Always use tools when appropriate.",
        )

    def run_api(self, query: str) -> str:
        """Run agent for API requests"""
        messages = [ChatMessage.from_user(query)]
        result = self.agent.run(messages=messages)
        return result["replies"][0].content

    async def run_chat_completion_async(
        self, model: str, messages: list[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        """Run agent for OpenAI-compatible chat completion"""
        chat_messages = [
            ChatMessage.from_openai_dict_format(message) for message in messages
        ]

        return async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={
                "messages": chat_messages,
            },
        )
```

## Agent Configuration Options

### System Prompt Configuration

```python
def setup(self) -> None:
    self.agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o"),
        system_prompt=self._get_system_prompt(),
        tools=[...],
    )

def _get_system_prompt(self) -> str:
    return """
    You are a helpful AI assistant with access to various tools.

    Guidelines:
    1. Be helpful and informative
    2. Use tools when appropriate
    3. If tools fail, explain why and suggest alternatives
    4. Always cite sources when using web search
    5. Be concise but thorough

    Available tools:
    - Web search: For current information
    - Calculator: For mathematical calculations
    """
```

### Tool Configuration

```python
def setup(self) -> None:
    # Configure individual tools
    search_tool = SerpDevTool(
        top_k=5,  # Number of results
        timeout=30,  # Timeout in seconds
    )

    calculator_tool = OpenAPITool(
        name="calculator",
        description="Perform mathematical calculations",
        timeout=10,
    )

    self.agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o"),
        tools=[search_tool, calculator_tool],
        system_prompt="...",
    )
```

## Agent Deployment Strategies

### 1. Production Agent

```python
from typing import AsyncGenerator, Dict, Any
from haystack.components.agents import Agent
from haystack.components.tools import SerpDevTool
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from hayhooks import BasePipelineWrapper, async_streaming_generator, log

class ProductionAgentWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Production-grade tools
        search_tool = SerpDevTool(
            top_k=5,
            timeout=30,
            country="us",
            language="en",
        )

        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(
                model="gpt-4o",
                temperature=0.7,
                max_tokens=2000,
            ),
            tools=[search_tool],
            system_prompt=self._get_production_system_prompt(),
        )

    def _get_production_system_prompt(self) -> str:
        return """
        You are a professional AI assistant with web search capabilities.

        Instructions:
        1. Always verify information with web search when asked about current events
        2. Provide accurate, up-to-date information
        3. Cite sources for all information obtained from web search
        4. Be professional and helpful
        5. If you cannot find information, admit it clearly

        Available tools:
        - Web search: Search for current information on any topic
        """

    async def run_chat_completion_async(
        self, model: str, messages: list[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        try:
            chat_messages = [
                ChatMessage.from_openai_dict_format(message) for message in messages
            ]

            log.info(f"Running agent with {len(chat_messages)} messages")

            return async_streaming_generator(
                pipeline=self.agent,
                pipeline_run_args={
                    "messages": chat_messages,
                },
            )
        except Exception as e:
            log.error(f"Agent execution failed: {e}")
            raise
```

### 2. Specialized Agent

```python
class ResearchAgentWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Research-specific tools
        search_tool = SerpDevTool(top_k=10)  # More results for research
        paper_tool = OpenAPITool(
            name="arxiv",
            description="Search academic papers",
            openapi_spec=...,  # ArXiv API spec
        )

        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o"),
            tools=[search_tool, paper_tool],
            system_prompt=self._get_research_system_prompt(),
        )

    def _get_research_system_prompt(self) -> str:
        return """
        You are a research assistant specializing in academic and technical research.

        Your capabilities:
        1. Search the web for current information
        2. Find academic papers and research
        3. Synthesize information from multiple sources
        4. Provide citations and references

        Guidelines:
        1. Be thorough and comprehensive
        2. Always verify information with multiple sources
        3. Provide proper citations
        4. Distinguish between facts and opinions
        5. Acknowledge limitations in your knowledge
        """
```

## Agent Development Tips

### 1. Tool Selection

Choose tools based on your agent's purpose:

- **General Purpose**: Web search, calculator, weather
- **Research**: Academic search, document retrieval
- **Business**: CRM, API integrations, data analysis
- **Technical**: Code search, documentation lookup

### 2. Error Handling

```python
async def run_chat_completion_async(
    self, model: str, messages: list[dict], body: dict
) -> AsyncGenerator[str, None]:
    try:
        chat_messages = [
            ChatMessage.from_openai_dict_format(message) for message in messages
        ]

        return async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={
                "messages": chat_messages,
            },
        )
    except Exception as e:
        log.error(f"Agent execution failed: {e}")
        # Return error message to user
        yield f"I apologize, but I encountered an error: {str(e)}"
```

### 3. Performance Optimization

```python
def setup(self) -> None:
    # Use appropriate model for your use case
    self.agent = Agent(
        chat_generator=OpenAIChatGenerator(
            model="gpt-3.5-turbo",  # Faster and cheaper for simple tasks
            streaming_callback=lambda x: None,
        ),
        tools=[...],
        system_prompt="...",
    )
```

### 4. Monitoring and Logging

```python
from hayhooks import log

class MonitoredAgentWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o"),
            tools=[...],
            system_prompt="...",
        )

    async def run_chat_completion_async(
        self, model: str, messages: list[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        start_time = time.time()
        chat_messages = [
            ChatMessage.from_openai_dict_format(message) for message in messages
        ]

        log.info(f"Agent request started with {len(chat_messages)} messages")

        try:
            result = async_streaming_generator(
                pipeline=self.agent,
                pipeline_run_args={
                    "messages": chat_messages,
                },
            )

            log.info(f"Agent request completed in {time.time() - start_time:.2f}s")
            return result
        except Exception as e:
            log.error(f"Agent request failed after {time.time() - start_time:.2f}s: {e}")
            raise
```

## Examples

### Customer Support Agent

```python
class CustomerSupportAgentWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Customer support tools
        knowledge_base_tool = OpenAPITool(
            name="knowledge_base",
            description="Search product documentation",
            openapi_spec=...,  # Knowledge base API
        )

        order_tool = OpenAPITool(
            name="orders",
            description="Look up order information",
            openapi_spec=...,  # Order management API
        )

        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o"),
            tools=[knowledge_base_tool, order_tool],
            system_prompt=self._get_support_system_prompt(),
        )

    def _get_support_system_prompt(self) -> str:
        return """
        You are a customer support assistant for our products.

        Your responsibilities:
        1. Help customers with product questions
        2. Look up order information
        3. Provide troubleshooting assistance
        4. Escalate complex issues when necessary

        Guidelines:
        1. Be empathetic and patient
        2. Use tools to find accurate information
        3. Provide clear, step-by-step instructions
        4. Always verify order information before sharing
        5. Know when to escalate to human agents
        """
```

### Code Assistant Agent

```python
class CodeAssistantAgentWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Code-related tools
        search_tool = SerpDevTool()  # For finding code examples
        documentation_tool = OpenAPITool(
            name="docs",
            description="Search programming documentation",
            openapi_spec=...,  # Documentation API
        )

        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o"),
            tools=[search_tool, documentation_tool],
            system_prompt=self._get_code_assistant_system_prompt(),
        )

    def _get_code_assistant_system_prompt(self) -> str:
        return """
        You are a code assistant that helps with programming tasks.

        Your capabilities:
        1. Write, debug, and explain code
        2. Find code examples and documentation
        3. Help with debugging and troubleshooting
        4. Explain programming concepts

        Guidelines:
        1. Write clean, well-commented code
        2. Explain your reasoning
        3. Provide multiple solutions when appropriate
        4. Include error handling and best practices
        5. Test your suggestions before providing them
        """
```

## Next Steps

- [PipelineWrapper](pipeline-wrapper.md) - Learn about wrapper implementation
- [Examples](../examples/overview.md) - See working examples
- [OpenAI Compatibility](../features/openai-compatibility.md) - Integrate with chat systems
