def normalize_function_tools(raw_tools: object) -> list[dict]:
    """
    Convert Responses API function tools to Chat Completions tool schema.

    Supports both shapes:
    - {"type":"function","name":"...","parameters":{...}}
    - {"type":"function","function":{"name":"...","parameters":{...}}}
    """
    if not isinstance(raw_tools, list):
        return []

    converted: list[dict] = []
    for tool in raw_tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        function_def = tool.get("function")
        if isinstance(function_def, dict):
            converted.append({"type": "function", "function": function_def})
            continue

        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue

        parameters = tool.get("parameters")
        function_spec: dict = {
            "name": name,
            "parameters": parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}},
        }
        description = tool.get("description")
        if isinstance(description, str) and description:
            function_spec["description"] = description
        strict = tool.get("strict")
        if isinstance(strict, bool):
            function_spec["strict"] = strict

        converted.append({"type": "function", "function": function_spec})
    return converted


def build_generation_kwargs(body: dict) -> dict:
    """Build Chat Completions generation kwargs from a Responses request body."""
    generation_kwargs: dict = {}

    tools = normalize_function_tools(body.get("tools"))
    if tools:
        generation_kwargs["tools"] = tools

    tool_choice = body.get("tool_choice")
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and isinstance(tool_choice.get("name"), str)
        and "function" not in tool_choice
    ):
        generation_kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice["name"]}}
    elif tool_choice is not None:
        generation_kwargs["tool_choice"] = tool_choice

    if body.get("parallel_tool_calls") is not None:
        generation_kwargs["parallel_tool_calls"] = body["parallel_tool_calls"]
    if body.get("temperature") is not None:
        generation_kwargs["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        generation_kwargs["top_p"] = body["top_p"]
    if body.get("max_output_tokens") is not None:
        generation_kwargs["max_completion_tokens"] = body["max_output_tokens"]

    return generation_kwargs


def client_tool_names(chat_tools: list[dict]) -> list[str]:
    """Extract function names from chat-completions tool definitions."""
    names: list[str] = []
    for tool in chat_tools:
        if not isinstance(tool, dict):
            continue
        function_def = tool.get("function")
        if not isinstance(function_def, dict):
            continue
        name = function_def.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names
