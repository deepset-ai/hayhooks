# Durable long-running A2A Agent

This example runs a Haystack Agent as a durable A2A task. Redis persists both
the execution checkpoints and the A2A task projection, so accepted work can
survive a Hayhooks restart.

This is a local reliability demonstration, not a production Redis or ingress
configuration. Before a production evaluation, apply the
[controlled beta deployment profile](../../docs/advanced/durable-execution-operations.md#controlled-beta-deployment-profile).

Run the setup commands from the repository root. The example requires `curl`
and `jq`. All durable examples use the same Compose service and Redis volume.

```bash
docker compose -f examples/durable-compose.yaml up -d
python -m pip install -e ".[durable,a2a]"

export OPENAI_API_KEY=...
export HAYHOOKS_DURABLE_REDIS_URL=redis://localhost:6379/0
export HAYHOOKS_A2A_TASK_STORE=auto
export HAYHOOKS_EXAMPLE_TOOL_DELAY_SECONDS=15
export HAYHOOKS_EXAMPLE_RECEIPT_DELAY_SECONDS=0
export HAYHOOKS_DURABLE_LEASE_DURATION_MS=5000
```

The five-second execution lease keeps the restart demonstration short. Active
workers heartbeat their claims; production deployments should tune this value
for their own workload and failure environment.

Open a first terminal and start Hayhooks in the foreground. It prints the PID
used for the forced-crash demonstration:

```bash
sh -c 'echo "Hayhooks PID: $$"; exec hayhooks a2a run --pipelines-dir examples/a2a_long_running/pipelines'
```

For a quick happy-path rehearsal, run the Rich demo client in a second
terminal. It creates unique message IDs, submits work, waits for approval,
approves it, and follows the task to completion:

```bash
python examples/a2a_long_running/demo.py
```

Use the manual requests below when presenting the protocol or demonstrating
crash recovery and cancellation.

In a second terminal, define a helper that sends the current request file and
prints the response:

```bash
send_a2a_request() {
  local request_file="$1"
  local response_file="${2:-a2a-res}"

  curl -fsS http://localhost:1418/long_running_agent/ \
    -H 'content-type: application/json' \
    -H 'A2A-Version: 1.0' \
    --data-binary @"$request_file" \
    --output "$response_file"

  jq . "$response_file"
}
```

Submit detached work and save the returned task ID:

```bash
jq -n '{
  "jsonrpc":"2.0",
  "id":"submit",
  "method":"SendMessage",
  "params":{
    "message":{
      "messageId":"prepare-demo",
      "role":"ROLE_USER",
      "parts":[{
        "text":"Prepare this document for indexing. document_id: hayhooks-guide. content: Hayhooks durable A2A work survives restarts."
      }]
    },
    "configuration":{"returnImmediately":true}
  }
}' > a2a-req

send_a2a_request a2a-req a2a-res

TASK_ID=$(jq -er '.result.task.id // .result.id' a2a-res)
```

Inspect the task with `GetTask`. Repeat these commands until its state is
`TASK_STATE_INPUT_REQUIRED`:

```bash
jq -n --arg id "$TASK_ID" \
  '{"jsonrpc":"2.0","id":"poll","method":"GetTask","params":{"id":$id}}' \
  > a2a-req

send_a2a_request a2a-req a2a-res
```

Approve the task with a follow-up A2A message. The persisted Agent checkpoint
already contains the original request:

```bash
jq -n --arg task_id "$TASK_ID" '{
  "jsonrpc":"2.0",
  "id":"resume",
  "method":"SendMessage",
  "params":{
    "message":{
      "messageId":"approval-demo",
      "taskId":$task_id,
      "role":"ROLE_USER",
      "parts":[{"text":"Approved; proceed."}]
    },
    "configuration":{"returnImmediately":true}
  }
}' > a2a-req

send_a2a_request a2a-req a2a-res
```

## Crash and replay

Run the `GetTask` commands again until progress contains “Indexing effect
committed.” The tool has written its SQLite row and remains open for 15 seconds.

In the second terminal, replace `<PID>` with the PID printed in the first
terminal to kill Hayhooks without graceful shutdown:

```bash
kill -9 <PID>
```

Return to the first terminal and run the same foreground command again against
the same Redis:

```bash
sh -c 'echo "Hayhooks PID: $$"; exec hayhooks a2a run --pipelines-dir examples/a2a_long_running/pipelines'
```

After about five seconds, run the same `GetTask` commands again. Redis reclaims
the interrupted execution and the Agent replays the tool from its previous
checkpoint. Continue inspecting until the task is terminal.

The indexing tool uses the execution ID and document ID as a SQLite primary
key. The first attempt reports `side_effect_applied: true`; replay reports
`false` instead of inserting the same effect twice. This demonstrates the
at-least-once contract: Hayhooks protects execution state, while external
effects still require application-level idempotency.

## Checkpoint efficiency: resume after indexing

The Agent follows a realistic ingestion workflow: it first cleans and splits
the document, then makes a small catalog receipt update. The durable
`after_tool` hook checkpoints the completed indexing result before the Agent
asks the model to call the receipt tool. That means a crash while the receipt
is running resumes from the indexed document instead of repeating the expensive
cleaning and splitting work.

For this demonstration, start a fresh task, approve it, and use no indexing
delay but a long receipt delay:

```bash
export HAYHOOKS_EXAMPLE_TOOL_DELAY_SECONDS=0
export HAYHOOKS_EXAMPLE_RECEIPT_DELAY_SECONDS=15
```

Wait until task progress contains “Indexing is checkpointed; holding the
lightweight receipt update,” then kill and restart Hayhooks as in the crash
replay section. The recovered Agent receives the saved indexing result and
continues with `publish_indexing_receipt`; it must not call
`prepare_document_for_indexing` again. The indexing table retains the example's
idempotency proof; the receipt tool has no irreversible effect, so replaying it
is safe.

## Cancellation

For a cancellation demonstration, submit and approve a fresh task, then send:

```bash
jq -n --arg id "$TASK_ID" \
  '{"jsonrpc":"2.0","id":"cancel","method":"CancelTask","params":{"id":$id}}' \
  > a2a-req

send_a2a_request a2a-req a2a-res
```

Cancellation is cooperative. A synchronous tool may finish its current work
before the Agent reaches the next cancellation checkpoint.

The SQLite database is a local effect-store demonstration. It defaults to the
operating system's temporary directory and survives a process restart on the
same machine. Set `HAYHOOKS_EXAMPLE_INDEX_DB` to a mounted path if the effect
must survive container replacement.

Press Ctrl-C in the first terminal to stop Hayhooks, then stop Redis. The static message IDs are intended
for a clean rehearsal; remove the volume before starting again:

```bash
docker compose -f examples/durable-compose.yaml down
# docker compose -f examples/durable-compose.yaml down -v
rm -f a2a-req a2a-res
```
