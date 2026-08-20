#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${HAYHOOKS_BASE_URL:-http://localhost:1416}"
BASE_URL="${BASE_URL%/}"
STATE="${TMPDIR:-/tmp}/hayhooks-resumable-stream-demo"
QUESTION="${QUESTION:-Write a detailed briefing of at least 2,500 words in 16 titled sections. Explain what Haystack does, how its pipelines and agents work, what Redis does, and how Redis can support durable execution. Include a comparison, an architecture walkthrough, a failure-and-recovery scenario, trade-offs, and a conclusion. Do not abbreviate.}"

command -v curl >/dev/null
command -v jq >/dev/null

STREAM_PATH="$(
  curl -fsS -X POST "$BASE_URL/chat_with_website/run-durable" \
    -H 'Content-Type: application/json' \
    -H "Idempotency-Key: curl-demo-$(date +%s)-$$" \
    -d "$(jq -nc --arg question "$QUESTION" '{question: $question}')" |
  jq -er '.links.stream'
)"
STREAM_URL="$BASE_URL$STREAM_PATH"
printf '%s\n' "$STREAM_URL" > "$STATE.url"
printf 'Streaming %s\nStop with Ctrl-C after at least 10 seconds.\n\n' "$STREAM_URL"

trap 'printf "\n"' EXIT
curl -NfsS "$STREAM_URL" |
  tee "$STATE.events" |
  awk '/^data: / {print substr($0, 7); fflush()}' |
  jq --unbuffered -jr '
    .payload.content? //
    if .status? then "\n\n[\(.status)]\n"
    elif .detail? then "\n\n[\(.detail)]\n"
    else empty
    end
  '
