# Yvann Longform Script+Audio MVP

This feature adds an additive long-form orchestration runner for Yvann "Images to Video" workflows.

## Added files

- `script_examples/longform_yvann_runner.py`
- `script_examples/longform_yvann_job_config.example.json`
- `custom_nodes/longform_yvann_launcher.py`
- `scripts/required_custom_nodes.json` (updated)

## What it does

- Validates environment (script/audio exists, ffmpeg/ffprobe, workflow template, custom nodes, API reachability).
- Uses audio as timeline master and creates chunk boundaries with optional overlap.
- Parses script and derives chunk-level scene summaries and prompts.
- Creates persistent job artifacts:
  - `job_config.json`
  - `job_state.json`
  - `manifest/chunk_manifest.json`
- Splits audio chunk-by-chunk to wav files without loading full audio in memory.
- Generates chunk images (backend selectable):
  - `comfy_api` (ComfyUI API text-to-image)
  - `procedural` (fallback deterministic abstract image generator)
- Converts Yvann full workflow JSON to API format via `/workflow/convert`.
- Injects chunk image/audio into `LoadImage` and `LoadAudio` nodes.
- Scales Yvann render workload per chunk using `yvann_render_fps`, `yvann_min_frames`, and `yvann_max_frames`.
- Executes per chunk with partial execution target set to selected `VHS_VideoCombine` output node.
- Saves each chunk video immediately and updates state after every chunk.
- Supports interruption/restart via manifest/state persistence.
- Optionally concatenates chunk videos with ffmpeg concat demuxer.

## ComfyUI launcher node

The repo now includes a custom node named `Yvann Longform Launcher` under the `Yvann/Longform` category.

It lets you launch jobs directly from a workflow using the ComfyUI server process, and it exposes job status at:

```text
/yvann_longform/jobs
```

The node writes a launch config next to the job outputs and starts the runner in the background.

## Run

From ComfyUI root:

```bash
python script_examples/longform_yvann_runner.py --config script_examples/longform_yvann_job_config.example.json
```

Dry-run mode (planning/chunking/image generation only):

```bash
python script_examples/longform_yvann_runner.py --config script_examples/longform_yvann_job_config.example.json --dry-run
```

## Output layout

For each run:

```text
<output_root>/job_YYYYMMDD_HHMMSS/
  job_config.json
  job_state.json
  script_source.txt
  <audio_source_name>
  manifest/
    chunk_manifest.json
  audio_chunks/
    chunk_0001.wav
  images/
    chunk_0001_img_01.png
  videos/
    chunk_0001.mp4
  previews/
  final/
    final_concat.mp4
```

## Notes

- Existing Yvann workflows are unchanged.
- This is additive and external to default short-form flows.
- For faster chunk turnaround, default output target is `First Pass | Low Res`.
- Switch `image_backend` to `procedural` if checkpoint-based image generation fails.
