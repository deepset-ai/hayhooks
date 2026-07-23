# Durable long-running A2A Agent

This executable example has two persistent projections:

- the durable execution record and checkpoints;
- the A2A protobuf task, owner, active index, projection version, and retention index.

Start Redis and the server:

```bash
docker compose -f examples/a2a_long_running/compose.yaml up -d
pip install "hayhooks[durable,a2a]"
export OPENAI_API_KEY=...
export HAYHOOKS_DURABLE_REDIS_URL=redis://localhost:6379/0
export HAYHOOKS_A2A_TASK_STORE=auto
hayhooks a2a run --pipelines-dir examples/a2a_long_running/pipelines
```

Submit detached work (`returnImmediately`) and save the returned task ID:

```bash
curl -s http://localhost:1418/long_running_agent/ \
  -H 'content-type: application/json' \
  -H 'A2A-Version: 1.0' \
  -d '{
    "jsonrpc":"2.0","id":"submit","method":"SendMessage",
    "params":{
      "message":{
        "messageId":"prepare-1","role":"ROLE_USER",
        "parts":[{"text":"Prepare document guide. Content: Hayhooks durable A2A work survives restarts."}]
      },
      "configuration":{"returnImmediately":true}
    }
  }' | tee /tmp/hayhooks-a2a-task.json

TASK_ID=$(jq -r '.result.task.id // .result.id' /tmp/hayhooks-a2a-task.json)
```

The first projection is `TASK_STATE_INPUT_REQUIRED`. Continue with only the new
message; the persisted Agent state already owns prior history:

```bash
curl -s http://localhost:1418/long_running_agent/ \
  -H 'content-type: application/json' -H 'A2A-Version: 1.0' \
  -d "{\"jsonrpc\":\"2.0\",\"id\":\"resume\",\"method\":\"SendMessage\",\"params\":{\"message\":{\"messageId\":\"approval-1\",\"taskId\":\"$TASK_ID\",\"role\":\"ROLE_USER\",\"parts\":[{\"text\":\"Approved; proceed.\"}]},\"configuration\":{\"returnImmediately\":true}}}"
```

Poll with `GetTask`:

```bash
watch -n 1 "curl -s http://localhost:1418/long_running_agent/ \
  -H 'content-type: application/json' -H 'A2A-Version: 1.0' \
  -d '{\"jsonrpc\":\"2.0\",\"id\":\"poll\",\"method\":\"GetTask\",\"params\":{\"id\":\"$TASK_ID\"}}' | jq"
```

Stop and restart Hayhooks while the three-second tool is running. Redis retains
both projections; a leased reconciler resumes task projection without allowing
a stale replica to overwrite a newer version.

Request cancellation:

```bash
curl -s http://localhost:1418/long_running_agent/ \
  -H 'content-type: application/json' -H 'A2A-Version: 1.0' \
  -d "{\"jsonrpc\":\"2.0\",\"id\":\"cancel\",\"method\":\"CancelTask\",\"params\":{\"id\":\"$TASK_ID\"}}"
```

The task first reports “Cancellation requested.” It becomes
`TASK_STATE_CANCELED` only after the execution record is terminal canceled.

The SQLite indexing row is an external effect. Its primary key combines
`current_execution_id()` with the logical indexing step, so replay returns the
existing effect instead of applying it twice. This illustrates the at-least-once
contract: Hayhooks fences state transitions, but every external effect still
needs application-level idempotency.
