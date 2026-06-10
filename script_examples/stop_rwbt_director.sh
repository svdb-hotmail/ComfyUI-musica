#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/workspace/ComfyUI}"
DIRECTOR_PID_PATH="${DIRECTOR_PID_PATH:-$COMFY_ROOT/output/rwbt_director_state/director.pid}"

if [[ ! -f "$DIRECTOR_PID_PATH" ]]; then
  echo "[INFO] No director pid file at $DIRECTOR_PID_PATH"
  exit 0
fi

pid="$(cat "$DIRECTOR_PID_PATH" 2>/dev/null || true)"
if [[ -z "$pid" ]]; then
  echo "[WARN] Empty pid file; removing it"
  rm -f "$DIRECTOR_PID_PATH"
  exit 0
fi

if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "[OK] Stopped director process $pid"
else
  echo "[INFO] Process $pid is not running"
fi

rm -f "$DIRECTOR_PID_PATH"
