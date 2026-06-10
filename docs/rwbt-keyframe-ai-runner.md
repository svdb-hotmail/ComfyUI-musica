# RWBT Keyframe AI Runner (Container-Local Only)

This runner generates ordered START/END keyframes from an RWBT markdown plan and can optionally use a local AI loop for prompt interpretation and consistency correction.

## Files

- Runner: `script_examples/rwbt_keyframe_ai_runner.py`
- Example config: `script_examples/rwbt_keyframe_ai_job_config.example.json`
- Custom node manifest: `scripts/required_custom_nodes.json` (includes `ComfyUI-Keyframed`)

## Local-Only Constraint

Use only models/services running on the same container:

- Image model defaults to a local SDXL base checkpoint target:
  - `sd_xl_base_1.0.safetensors`
  - If not installed yet, copy it into `models/checkpoints/` and keep `model_name` aligned.
- Comfy endpoint is local:
  - `http://127.0.0.1:18188`
- Optional AI endpoint should also be local (for example Ollama):
  - `http://127.0.0.1:11434/v1`

## 1) Sync required custom nodes

```bash
cd /workspace/ComfyUI
python scripts/sync_required_custom_nodes.py --comfy-root /workspace/ComfyUI --manifest /workspace/ComfyUI/scripts/required_custom_nodes.json --mode sync
```

## 2) Prepare workflow template

Create/export a still-image workflow JSON to:

- `user/default/workflows/rwbt_sdxl_keyframed_template.json`

The runner auto-patches common node types (`CLIPTextEncode`, `KSampler`, `CheckpointLoaderSimple`, `SaveImage`, etc.).
If your graph is custom, add `workflow_overrides` entries in config to target specific nodes/titles/widgets.

For variable image references per prompt:

- Keep reference loaders in your workflow as `LoadImage` nodes.
- Name them with keywords like `Reference`, `Ref`, or `IPAdapter`.
- The runner resolves references from prompt text (`Previously generated ...`, `RWBT (n)`) and assigns them per task.

## 3) Use example config

Copy and edit:

- `script_examples/rwbt_keyframe_ai_job_config.example.json`

Required minimum:

- `prompt_plan_path`
- `workflow_template_path`
- `output_root`

## 4) Dry-run

```bash
cd /workspace/ComfyUI
python script_examples/rwbt_keyframe_ai_runner.py --config script_examples/rwbt_keyframe_ai_job_config.example.json --dry-run
```

## 5) Full run

```bash
cd /workspace/ComfyUI
python script_examples/rwbt_keyframe_ai_runner.py --config script_examples/rwbt_keyframe_ai_job_config.example.json
```

## Optional local AI loop

Enable in config:

- `ai.enabled: true`
- `ai.api_base: "http://127.0.0.1:11434/v1"`
- `ai.model: "qwen2.5vl:7b"` (or another local model)

If using an OpenAI-compatible local service that requires a token, set `OPENAI_API_KEY` inside the container.

## Outputs

Each run creates:

- `output/rwbt_jobs/<job_id>/images/` keyframe images
- `output/rwbt_jobs/<job_id>/manifest/tasks_manifest.json`
- `output/rwbt_jobs/<job_id>/job_state.json`
