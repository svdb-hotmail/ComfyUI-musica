# Yvann Longform Script+Audio MVP

This feature adds an additive long-form orchestration runner for Yvann "Images to Video" workflows.

## Added files

- `script_examples/longform_yvann_runner.py`
- `script_examples/longform_yvann_job_config.example.json`
- `custom_nodes/longform_yvann_launcher.py`
- `scripts/required_custom_nodes.json` (updated)
- `script_examples/workflows/AudioReactive_ImagesToVideo_Yvann (Longform Launcher).json`
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
- Converts Yvann full workflow JSON to API format via `/workflow/convert`.
- Injects the generated scene image folder and chunk audio into the Yvann workflow.
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

This copies the tracked longform-enhanced template into the Yvann base workflow path.

```bash
python scripts/apply_yvann_longform_template.py --repo-root . --apply-user-workflow
```

Target paths updated by this command:

- `custom_nodes/comfyui_yvann-nodes/example_workflows/AudioReactive_ImagesToVideo_Yvann.json`
- `user/default/workflows/AudioReactive_ImagesToVideo_Yvann.json` (when `--apply-user-workflow` is set)

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

- Existing Yvann workflows are unchanged.
- This is additive and external to default short-form flows.
- For faster chunk turnaround, default output target is `First Pass | Low Res`.
- Switch `image_backend` to `procedural` if checkpoint-based image generation fails.
