#!/usr/bin/env bash

set -euo pipefail

STATE="${TMPDIR:-/tmp}/hayhooks-resumable-stream-demo"
command -v curl >/dev/null
command -v jq >/dev/null

if [[ ! -s "$STATE.url" || ! -s "$STATE.events" ]]; then
  printf 'Run start_stream.sh first, then stop it with Ctrl-C.\n' >&2
  exit 1
fi

STREAM_URL="$(<"$STATE.url")"
LAST_EVENT_ID="$(
  awk '
    {sub(/\r$/, "")}
    /^id: / {candidate = substr($0, 5)}
    /^$/ && candidate != "" {last = candidate; candidate = ""}
    END {print last}
  ' "$STATE.events"
)"

if [[ -z "$LAST_EVENT_ID" ]]; then
  printf 'No complete streamed event is available to resume yet.\n' >&2
  exit 1
fi

printf 'Resuming after Last-Event-ID: %s\n\n' "$LAST_EVENT_ID"
curl -NfsS -H "Last-Event-ID: $LAST_EVENT_ID" "$STREAM_URL" |
  awk '/^data: / {print substr($0, 7); fflush()}' |
  jq --unbuffered -jr '
    .payload.content? //
    if .status? then "\n\n[\(.status)]\n"
    elif .detail? then "\n\n[\(.detail)]\n"
    else empty
    end
  '
