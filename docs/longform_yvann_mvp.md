# Yvann Longform Script+Audio MVP

This feature adds a minimal long-form orchestration runner for the Yvann "Images to Video" workflow.

## Added files

- `script_examples/longform_yvann_runner.py`
- `script_examples/longform_yvann_job_config.example.json`
- `custom_nodes/longform_yvann_launcher.py`
- `scripts/required_custom_nodes.json` (updated)
- `script_examples/workflows/AudioReactive_ImagesToVideo_Yvann (Longform Launcher).json` (the compact dashboard installed into `user/default/workflows`)
- `scripts/apply_yvann_longform_template.py`

## What it does

- Validates environment (script/audio exists, ffmpeg/ffprobe, workflow template, custom nodes, API reachability).
- Uses audio as timeline master and creates chunk boundaries with optional overlap.
- Parses script/cue sheets and derives chunk-level scene summaries and prompts.
- Supports DJ-mix cue sheets with visual switch markers like `# A. 00:00:00 Rocket preparing for launch`.
- Splits render chunks at visual marker timestamps so a chunk never crosses into the next visual batch.
- Creates persistent job artifacts:
  - `job_config.json`
  - `job_state.json`
  - `manifest/chunk_manifest.json`
- Splits audio chunk-by-chunk to wav files without loading full audio in memory.
- Generates timestamp scene image folders (backend selectable):
  - `comfy_api` (ComfyUI API text-to-image)
  - `procedural` (fallback deterministic abstract image generator)
- Generates one varied image every `image_interval_seconds` for each cue/timestamp span.
- Feeds the whole generated scene folder into Yvann through a standard KJNodes folder image-batch loader (`LoadImagesFromFolderKJ`).
- Converts the Yvann Images-to-Video workflow JSON to API format via `/workflow/convert`.
- Injects the generated scene image folder and chunk audio into the Yvann workflow.
- Scales Yvann render workload per chunk using `yvann_render_fps`, `yvann_min_frames`, and `yvann_max_frames`.
- Executes per chunk with partial execution target set to selected `VHS_VideoCombine` output node.
- Saves each chunk video immediately and updates state after every chunk.
- Supports interruption/restart via manifest/state persistence.
- Optionally concatenates chunk videos with ffmpeg concat demuxer.

## ComfyUI launcher node

The repo now includes the core longform nodes under the `Yvann/Longform` category:

- `Yvann Longform Audio Source`
- `Yvann Longform Cue Sheet`
- `Yvann Longform Render Profile`
- `Yvann Longform Execution Settings`
- `Yvann Longform Image-to-Video`
- `Yvann Longform Launcher`
- `Yvann Longform Audio Analysis Preview`
- `Yvann Longform Batch Plan`
- `Yvann Longform Job Status`
- `Yvann Longform Cancel Job`
- `Yvann Longform Generated Images`
- `Yvann Longform Scene Batch`
- `Yvann Workflow Inspector`

It lets you launch jobs directly from a workflow using the ComfyUI server process. The workflow acts as a compact dashboard for backend work: closing or refreshing the browser does not stop the runner. Audio selection/upload, cue-sheet text, render profile, and Yvann execution settings are separate linked nodes that feed the job launcher and preview nodes. The preview therefore reads the same cue sheet the backend job will launch with, instead of carrying its own duplicate text. The full Yvann render graph is not opened as the dashboard; it stays hidden under `custom_nodes/comfyui_yvann-nodes/example_workflows/AudioReactive_ImagesToVideo_Yvann.json` and is loaded by the backend runner.

The audio source node intentionally starts without a fixed sample path in the dashboard. Select an uploaded/local `AUDIO` input, choose an existing file under `input/`, or type a path explicitly before launching.

It exposes persisted job status at:

```text
/yvann_longform/jobs
```

The node writes a launch config and process record next to the job outputs and starts the runner in the background. The dashboard reads backend-owned `job_state.json`, `manifest/chunk_manifest.json`, and process records.

Cancellation is requested through the backend at:

```text
POST /yvann_longform/jobs/<job_id>/cancel
```

or by queuing `Yvann Longform Cancel Job`. Cancellation writes `cancel.requested` into the job folder. The runner checks that marker between stages and while waiting for ComfyUI prompts, interrupts active prompts, and marks the job cancelled.

## Run

From ComfyUI root:

```bash
python script_examples/longform_yvann_runner.py --config script_examples/longform_yvann_job_config.example.json
```

## Cue-sheet input format

For long mixes, keep the music track list and add visual switch markers in comments:

```text
00:00:00  1 Artist - Track Title                  # A. 00:00:00 Rocket preparing for launch
00:04:39  2 Artist - Next Track                   # B. 00:03:30 Rocket taking off, stage separation
00:07:49  3 Artist - Another Track                #
00:13:25  4 Artist - Later Track                  # C. 00:11:30 Satellite images of Earth and planets
```

The timestamp after the cue label controls when that visual batch starts. Continuation comment lines without a timestamp are appended to the previous visual batch prompt. The runner generates a folder of varied images for each timestamp span, at the configured cadence such as one image every 5 seconds. It still renders in manageable audio/video chunks, but each chunk points Yvann at the full active scene folder through the folder batch loader.

See:

```text
script_examples/longform_yvann_cuesheet.example.txt
```

## Apply the longform-modified base Yvann template

This copies the tracked compact dashboard into `user/default/workflows/AudioReactive_ImagesToVideo_Yvann.json`. It intentionally installs only the longform dashboard and avoids the older all-in-one, preview, inspection, and raw render-graph canvases.

```bash
python scripts/apply_yvann_longform_template.py --repo-root . --apply-user-workflow
```

Target path updated by this command:

- `user/default/workflows/AudioReactive_ImagesToVideo_Yvann.json` (when `--apply-user-workflow` is set)

## Apply on a new RTX 6000PRO container

After pulling this fork on a fresh container, run:

```bash
bash scripts/apply_rtx6000pro_longform_optimizations.sh
```

The script validates the runner/custom nodes, applies the single compact dashboard workflow, removes stale unused workflow copies, sets ComfyUI launch args for `--highvram` and `--enable-triton-backend`, then restarts the supervisor `comfyui` process when available. SageAttention is not enabled by default because this workflow can hit unsupported attention head dimensions and fall back noisily to PyTorch attention.

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
    A/
      A_0001.png
      A_0002.png
      ...
  videos/
    chunk_0001.mp4
  previews/
  final/
    final_concat.mp4
```

## Notes

- Only the longform Images-to-Video path is installed by the template script.
- The final concatenated MP4 is normalized to 1280x720 at 24fps.
- For faster chunk turnaround, default output target is `First Pass | Low Res`.
- Switch `image_backend` to `procedural` if checkpoint-based image generation fails.
