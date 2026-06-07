#!/usr/bin/env bash
set -euo pipefail

COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
SUPERVISOR_PROGRAM="${SUPERVISOR_PROGRAM:-comfyui}"
COMFYUI_ARGS_VALUE="${COMFYUI_ARGS_VALUE:---disable-auto-launch --port 18188 --enable-cors-header --highvram --enable-triton-backend}"

if [[ ! -d "$COMFYUI_DIR" ]]; then
  echo "ComfyUI directory not found: $COMFYUI_DIR" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

cd "$COMFYUI_DIR"

"$PYTHON_BIN" -m py_compile \
  script_examples/longform_yvann_runner.py \
  custom_nodes/longform_yvann_launcher.py \
  scripts/apply_yvann_longform_template.py

"$PYTHON_BIN" scripts/apply_yvann_longform_template.py --repo-root "$COMFYUI_DIR" --apply-user-workflow

rm -f \
  user/default/workflows/"AudioReactive_ImagesToVideo_Yvann (Longform Launcher).json" \
  user/default/workflows/AudioReactive_ImagesToVideo_Yvann_Longform_Batch_Generator.json \
  user/default/workflows/Yvann_Longform_All_In_One_2H_Video_WORKFLOW.json \
  user/default/workflows/Yvann_Longform_Batch_Generator_RUN_ME.json \
  user/default/workflows/Yvann_Longform_ComfyUI_Workflow_RUN_THIS.json \
  script_examples/workflows/Yvann_Longform_All_In_One_2H_Video_WORKFLOW.json \
  script_examples/workflows/Yvann_Longform_Batch_Generator_RUN_ME.json \
  script_examples/workflows/Yvann_Longform_ComfyUI_Workflow_RUN_THIS.json

if [[ -w /etc/environment ]]; then
  cp /etc/environment "/etc/environment.comfy-before-longform-optimizations.$(date +%Y%m%d_%H%M%S)"
  "$PYTHON_BIN" - <<PY
from pathlib import Path
path = Path('/etc/environment')
line = 'COMFYUI_ARGS="${COMFYUI_ARGS_VALUE}"'
lines = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
for index, existing in enumerate(lines):
    if existing.startswith('COMFYUI_ARGS='):
        lines[index] = line
        break
else:
    lines.append(line)
path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
else
  echo "Skipping /etc/environment update; file is not writable." >&2
fi

if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart "$SUPERVISOR_PROGRAM"
else
  echo "supervisorctl not found; restart ComfyUI manually to apply COMFYUI_ARGS." >&2
fi

echo "Applied longform image-to-video dashboard, backend dashboard/cancel support, and RTX 6000PRO runtime flags."
echo "COMFYUI_ARGS=$COMFYUI_ARGS_VALUE"