#!/usr/bin/env bash
#
# Generate dashboard traces using real pipeline wrappers from examples/.
#
# Prerequisites:
#   - Hayhooks running on localhost:1416 with dashboard enabled
#   - OPENAI_API_KEY set (for LLM-based pipelines)
#   - jq installed
#
# Usage:
#   bash scripts/demo_dashboard_traces.sh            # 0.3s delay (fast — generates all traces, then explore manually)
#   DELAY=2 bash scripts/demo_dashboard_traces.sh    # 2s delay (watch traces appear live)
#
set -euo pipefail

BASE="${HAYHOOKS_BASE_URL:-http://localhost:1416}"
DELAY="${DELAY:-0.3}"
EXAMPLES="$(cd "$(dirname "$0")/../examples/pipeline_wrappers" && pwd)"
HAS_KEY="${OPENAI_API_KEY:+true}"
HAS_KEY="${HAS_KEY:-false}"

blue()  { printf '\033[1;34m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
dim()   { printf '\033[2m%s\033[0m\n' "$*"; }
step()  { echo; blue "── $1"; }
pause() { sleep "$DELAY"; }

request() {
  local code body
  body=$(curl -s -w '\n%{http_code}' "$@")
  code=$(echo "$body" | tail -1)
  body=$(echo "$body" | sed '$d')
  echo "$code"
  echo "$body" >&2
}

deploy_files() {
  local name="$1" overwrite="${2:-false}"
  shift 2
  local files_json="{}"
  while [ $# -gt 0 ]; do
    local filepath="$1"; shift
    local filename
    filename=$(basename "$filepath")
    files_json=$(jq --arg name "$filename" --arg content "$(cat "$filepath")" \
      '. + {($name): $content}' <<< "$files_json")
  done
  local payload
  payload=$(jq -n \
    --arg name "$name" \
    --argjson files "$files_json" \
    --argjson overwrite "$overwrite" \
    '{name:$name, files:$files, overwrite:$overwrite, save_files:false}')
  request -X POST "$BASE/deploy_files" -H 'Content-Type: application/json' -d "$payload"
}

check() {
  local expected="$1" actual="$2" ok_msg="$3" fail_msg="$4"
  if [ "$actual" = "$expected" ]; then green "  ✓ $ok_msg"; else red "  ✗ $fail_msg ($actual)"; fi
}

blue "╔══════════════════════════════════════════╗"
blue "║     Hayhooks Dashboard Demo Script      ║"
blue "╚══════════════════════════════════════════╝"
echo

if [ "$HAS_KEY" = false ]; then
  dim "OPENAI_API_KEY not set — LLM pipelines will be skipped"
fi

# ===================================================================
# PHASE 1 — DEPLOY
# ===================================================================
blue "── PHASE 1: Deploy pipelines ────────────────────"
echo

step "1) Deploy 'calculator'"
CODE=$(deploy_files calculator true \
  "$EXAMPLES/relative_imports/pipeline_wrapper.py" \
  "$EXAMPLES/relative_imports/utils.py")
check 200 "$CODE" "deployed calculator" "deploy failed"
pause

step "2) Deploy 'calculator' again WITHOUT overwrite → expect 409"
CODE=$(deploy_files calculator false \
  "$EXAMPLES/relative_imports/pipeline_wrapper.py" \
  "$EXAMPLES/relative_imports/utils.py")
check 409 "$CODE" "correctly rejected (409)" "expected 409"
pause

step "3) Deploy 'question_answer'"
QA_OK=false
if [ "$HAS_KEY" = true ]; then
  CODE=$(deploy_files question_answer true \
    "$EXAMPLES/async_question_answer/pipeline_wrapper.py" \
    "$EXAMPLES/async_question_answer/question_answer.yml")
  check 200 "$CODE" "deployed question_answer" "deploy failed"
  [ "$CODE" = "200" ] && QA_OK=true
else
  dim "  skipped (no OPENAI_API_KEY)"
fi
pause

step "4) Deploy 'hybrid_streaming'"
HS_OK=false
if [ "$HAS_KEY" = true ]; then
  CODE=$(deploy_files hybrid_streaming true \
    "$EXAMPLES/async_hybrid_streaming/pipeline_wrapper.py" \
    "$EXAMPLES/async_hybrid_streaming/hybrid_streaming.yml")
  check 200 "$CODE" "deployed hybrid_streaming" "deploy failed"
  [ "$CODE" = "200" ] && HS_OK=true
else
  dim "  skipped (no OPENAI_API_KEY)"
fi
pause

step "5) Deploy 'reasoning_agent'"
RA_OK=false
if [ "$HAS_KEY" = true ]; then
  CODE=$(deploy_files reasoning_agent true \
    "$EXAMPLES/reasoning_agent/pipeline_wrapper.py")
  check 200 "$CODE" "deployed reasoning_agent" "deploy failed"
  [ "$CODE" = "200" ] && RA_OK=true
else
  dim "  skipped (no OPENAI_API_KEY)"
fi
pause

# ===================================================================
# PHASE 2 — GENERATE TRACES
# ===================================================================
echo
echo
blue "╔══════════════════════════════════════════════════════════╗"
blue "║                                                          ║"
blue "║   Open $BASE/dashboard   ║"
blue "║                                                          ║"
blue "╚══════════════════════════════════════════════════════════╝"
echo
dim "  Traces will appear in real time…"
sleep 2
echo

blue "── PHASE 2: Generate traces ────────────────────"
echo

# ---- calculator (REST ×3, OpenAI 501) ----
step "6) Run 'calculator' via REST (×3 requests to build up trace list)"
for i in 1 2 3; do
  BODY=$(curl -s -X POST "$BASE/calculator/run" \
    -H 'Content-Type: application/json' \
    -d '{"name":"Hayhooks","numbers":[10,20,30,40]}')
  green "  ✓ #$i: $BODY"
  pause
done

step "7) Run 'calculator' via OpenAI → expect 501 (no chat support)"
CODE=$(request -X POST "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"calculator","messages":[{"role":"user","content":"Hello"}],"stream":false}')
check 501 "$CODE" "correctly unsupported for OpenAI chat (501)" "unexpected status"
pause

# ---- question_answer ----
step "8) Run 'question_answer' via REST (×2)"
if [ "$QA_OK" = true ]; then
  for i in 1 2; do
    BODY=$(curl -s --max-time 30 -X POST "$BASE/question_answer/run" \
      -H 'Content-Type: application/json' \
      -d '{"question":"What is the capital of France?"}')
    green "  ✓ REST #$i: $(echo "$BODY" | head -c 200)"
    pause
  done
else
  dim "  skipped"
  pause
fi

step "9) Run 'question_answer' via OpenAI (non-streaming)"
if [ "$QA_OK" = true ]; then
  BODY=$(curl -s --max-time 30 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"question_answer","messages":[{"role":"user","content":"What is Haystack by deepset?"}],"stream":false}')
  green "  ✓ OpenAI: $(echo "$BODY" | jq -r '.choices[0].message.content // .detail // .' 2>/dev/null | head -c 200)"
else
  dim "  skipped"
fi
pause

step "10) Run 'question_answer' via OpenAI (streaming)"
if [ "$QA_OK" = true ]; then
  dim "  streaming response:"
  curl -s -N --max-time 30 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"question_answer","messages":[{"role":"user","content":"Summarise Haystack in three bullet points."}],"stream":true}' \
    | while IFS= read -r line; do
        line="${line#data: }"
        [ -z "$line" ] || [ "$line" = "[DONE]" ] && continue
        chunk=$(echo "$line" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
        [ -n "$chunk" ] && printf '%s' "$chunk"
      done
  echo
  green "  ✓ streaming complete"
else
  dim "  skipped"
fi
pause

# ---- hybrid_streaming ----
step "11) Run 'hybrid_streaming' via REST"
if [ "$HS_OK" = true ]; then
  BODY=$(curl -s --max-time 30 -X POST "$BASE/hybrid_streaming/run" \
    -H 'Content-Type: application/json' \
    -d '{"question":"Explain quantum computing in one sentence."}')
  green "  ✓ REST result: $(echo "$BODY" | head -c 200)"
else
  dim "  skipped"
fi
pause

step "12) Run 'hybrid_streaming' via OpenAI (SSE streaming)"
if [ "$HS_OK" = true ]; then
  dim "  streaming response:"
  curl -s -N --max-time 30 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"hybrid_streaming","messages":[{"role":"user","content":"What is Haystack?"}],"stream":true}' \
    | while IFS= read -r line; do
        line="${line#data: }"
        [ -z "$line" ] || [ "$line" = "[DONE]" ] && continue
        chunk=$(echo "$line" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
        [ -n "$chunk" ] && printf '%s' "$chunk"
      done
  echo
  green "  ✓ streaming complete"
else
  dim "  skipped"
fi
pause

# ---- reasoning_agent ----
step "13) Run 'reasoning_agent' via OpenAI (non-streaming)"
if [ "$RA_OK" = true ]; then
  BODY=$(curl -s --max-time 30 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"reasoning_agent","messages":[{"role":"user","content":"Why is the sky blue? Answer briefly."}],"stream":false}')
  green "  ✓ OpenAI: $(echo "$BODY" | jq -r '.choices[0].message.content // .detail // .' 2>/dev/null | head -c 200)"
else
  dim "  skipped"
fi
pause

step "14) Run 'reasoning_agent' via OpenAI (streaming)"
if [ "$RA_OK" = true ]; then
  dim "  streaming response:"
  curl -s -N --max-time 30 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"reasoning_agent","messages":[{"role":"user","content":"Summarise the history of quantum mechanics in three sentences."}],"stream":true}' \
    | while IFS= read -r line; do
        line="${line#data: }"
        [ -z "$line" ] || [ "$line" = "[DONE]" ] && continue
        chunk=$(echo "$line" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
        [ -n "$chunk" ] && printf '%s' "$chunk"
      done
  echo
  green "  ✓ streaming complete"
else
  dim "  skipped"
fi
pause

# ===================================================================
# PHASE 3 — CLEANUP
# ===================================================================
echo
blue "── PHASE 3: Cleanup ─────────────────────────────"
echo

step "15) Undeploy 'calculator'"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/undeploy/calculator")
check 200 "$CODE" "undeployed calculator" "undeploy failed"

echo
green "Done — all traces visible at $BASE/dashboard"
