# A2A Multi-Agent Demo

Two Haystack agents run behind one `hayhooks a2a run` server:

- `trip_planner_agent` receives the user request, calls `weather_agent` over A2A, then calls its own activities MCP tool.
- `weather_agent` answers weather questions using its own Open-Meteo MCP tool.

This keeps the split clear: A2A is agent-to-agent delegation; MCP is agent-to-tool access.

```text
weather MCP (:8001)                  activities MCP (:8002)
        ^                                    ^
        | MCP                                | MCP
weather_agent <------ A2A ------ trip_planner_agent
        \____________ hayhooks a2a :1418 ____________/
```

## Run It

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python run_demo.py
```

The script starts both MCP servers, starts Hayhooks A2A on port `1418`, discovers both agent cards, sends one trip-planning request, streams the answer, then keeps the servers running until Ctrl-C.

Use a custom question:

```bash
python run_demo.py "I'm in Lisbon today, what should I do?"
```

Start only the servers for manual testing:

```bash
python run_demo.py --serve
```

## Try It Manually

```bash
# Chat with the trip planner. It should call weather_agent over A2A.
python chat.py

# Chat with the weather agent directly.
python chat.py weather_agent

# Inspect generated agent cards.
curl -s http://localhost:1418/weather_agent/.well-known/agent-card.json | jq
curl -s http://localhost:1418/trip_planner_agent/.well-known/agent-card.json | jq

# Send one A2A 1.0 task.
curl -s http://localhost:1418/weather_agent/ \
  -H "Content-Type: application/json" -H "A2A-Version: 1.0" \
  -d '{"jsonrpc":"2.0","id":"1","method":"SendMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","parts":[{"text":"Weather in Berlin?"}]}}}' | jq
```

The server also accepts A2A 0.3 JSON-RPC methods by default (`HAYHOOKS_A2A_V0_3_COMPAT=true`).

## Configuration

Useful environment variables:

- `HAYHOOKS_A2A_PORT` defaults to `1418`
- `A2A_SERVER_URL` defaults to `http://localhost:1418`
- `WEATHER_AGENT_A2A_URL` defaults to `http://localhost:1418/weather_agent`
- `WEATHER_MCP_URL` defaults to `http://localhost:8001/mcp`
- `ACTIVITIES_MCP_URL` defaults to `http://localhost:8002/mcp`

Agent cards are customized in each `pipeline_wrapper.py` via `a2a_card`. Because this demo hosts multiple agents on one server, cards live under `/{agent}/.well-known/agent-card.json`; run one A2A server per agent if a client requires root-level discovery.
