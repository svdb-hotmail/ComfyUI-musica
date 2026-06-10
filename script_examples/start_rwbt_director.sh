#!/usr/bin/env bash
set -euo pipefail

# Starts the persistent RWBT director service in the background.
# The service survives ComfyUI page reloads and stores plan/session memory on disk.

COMFY_ROOT="${COMFY_ROOT:-/workspace/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DIRECTOR_HOST="${DIRECTOR_HOST:-127.0.0.1}"
DIRECTOR_PORT="${DIRECTOR_PORT:-8099}"
DIRECTOR_UPSTREAM_API_BASE="${DIRECTOR_UPSTREAM_API_BASE:-http://127.0.0.1:8000/v1}"
DIRECTOR_MODEL="${DIRECTOR_MODEL:-Qwen/Qwen3.5-9B}"
DIRECTOR_SESSION_ID="${DIRECTOR_SESSION_ID:-rwbt-main}"
DIRECTOR_STATE_PATH="${DIRECTOR_STATE_PATH:-$COMFY_ROOT/output/rwbt_director_state/state.json}"
DIRECTOR_LOG_PATH="${DIRECTOR_LOG_PATH:-$COMFY_ROOT/output/rwbt_director_state/director.log}"
DIRECTOR_PID_PATH="${DIRECTOR_PID_PATH:-$COMFY_ROOT/output/rwbt_director_state/director.pid}"

mkdir -p "$(dirname "$DIRECTOR_STATE_PATH")"

if [[ -f "$DIRECTOR_PID_PATH" ]]; then
  old_pid="$(cat "$DIRECTOR_PID_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[INFO] Director already running with PID $old_pid"
    exit 0
  fi
fi

nohup "$PYTHON_BIN" "$COMFY_ROOT/script_examples/rwbt_director_server.py" \
  --host "$DIRECTOR_HOST" \
  --port "$DIRECTOR_PORT" \
  --upstream-api-base "$DIRECTOR_UPSTREAM_API_BASE" \
  --model "$DIRECTOR_MODEL" \
  --default-session-id "$DIRECTOR_SESSION_ID" \
  --state-path "$DIRECTOR_STATE_PATH" \
  > "$DIRECTOR_LOG_PATH" 2>&1 &

pid=$!
echo "$pid" > "$DIRECTOR_PID_PATH"

echo "[OK] RWBT director started"
echo "     PID: $pid"
echo "     API: http://$DIRECTOR_HOST:$DIRECTOR_PORT/v1/chat/completions"
echo "  Health: http://$DIRECTOR_HOST:$DIRECTOR_PORT/health"
echo "    Log: $DIRECTOR_LOG_PATH"
