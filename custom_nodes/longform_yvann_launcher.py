from __future__ import annotations

import json
import os
import wave
import subprocess
import sys
import time
from pathlib import Path
import re

from aiohttp import web
import torch
import numpy as np
from PIL import Image

from server import PromptServer


JOB_REGISTRY: dict[str, dict[str, object]] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value.strip())
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _job_id() -> str:
    return f"job_{time.strftime('%Y%m%d_%H%M%S')}"


def _config_path(output_root: Path, job_id: str) -> Path:
    return output_root / "_launcher_configs" / f"{job_id}.json"


def _runner_path() -> Path:
    return _repo_root() / "script_examples" / "longform_yvann_runner.py"


def _hms_to_sec(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported timestamp format: {value}")


def _sec_to_hms(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 3600:02d}:{(whole % 3600) // 60:02d}:{whole % 60:02d}"


def _audio_duration(audio_path: Path) -> float | None:
    if not audio_path.exists():
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        return float(proc.stdout.strip())
    except Exception:
        return None


def _extract_cues(cue_sheet_text: str, total_duration: float | None = None) -> list[dict[str, object]]:
    marker_pattern = re.compile(
        r"#\s*(?:(?P<label>[A-Z])\.\s*)?(?P<time>\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s*(?P<text>.*)$",
        re.IGNORECASE,
    )
    continuation_pattern = re.compile(r"^\s*#\s*(?P<text>.+?)\s*$")
    markers: list[tuple[str, float, list[str]]] = []
    for line in str(cue_sheet_text).splitlines():
        marker = marker_pattern.search(line)
        if marker:
            label = (marker.group("label") or f"cue_{len(markers) + 1:02d}").upper()
            start_time = _hms_to_sec(marker.group("time"))
            text = marker.group("text").strip()
            markers.append((label, start_time, [text] if text else []))
            continue
        continuation = continuation_pattern.match(line)
        if continuation and markers:
            text = continuation.group("text").strip()
            if text:
                markers[-1][2].append(text)

    fallback_end = total_duration
    if fallback_end is None and markers:
        fallback_end = markers[-1][1] + 45.0
    cues: list[dict[str, object]] = []
    for idx, (label, start_time, parts) in enumerate(markers):
        next_start = fallback_end or (start_time + 45.0)
        for _next_label, candidate_start, _next_parts in markers[idx + 1:]:
            if candidate_start > start_time:
                next_start = candidate_start
                break
        if fallback_end is not None:
            next_start = min(next_start, fallback_end)
        summary = " ".join(" ".join(parts).split())
        if summary and next_start > start_time:
            cues.append({"id": label, "start": start_time, "end": next_start, "summary": summary})
    return cues


def _plan_chunks(total_duration: float, cue_starts: list[float], chunk_duration: float, max_chunks: int) -> list[tuple[float, float]]:
    cue_splits = sorted({round(p, 3) for p in cue_starts if 0.0 < p < total_duration})
    chunks: list[tuple[float, float]] = []
    t = 0.0
    while t < total_duration:
        end = min(total_duration, t + max(1.0, float(chunk_duration)))
        for split in cue_splits:
            if t + 0.001 < split < end - 0.001:
                end = split
                break
        chunks.append((t, end))
        if max_chunks > 0 and len(chunks) >= max_chunks:
            break
        if end >= total_duration:
            break
        t = end
    return chunks


def _latest_job_dir(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    jobs = sorted([p for p in output_root.glob("job_*") if p.is_dir()], reverse=True)
    return jobs[0] if jobs else None


def _latest_manifest_job_dir(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    jobs = sorted([p for p in output_root.glob("job_*") if p.is_dir()], reverse=True)
    for job in jobs:
        if (job / "manifest" / "chunk_manifest.json").exists():
            return job
    return jobs[0] if jobs else None


def _image_ui_entry(path: Path) -> dict[str, str] | None:
    output_root = _repo_root() / "output"
    try:
        rel = path.resolve().relative_to(output_root.resolve())
    except Exception:
        return None
    return {"filename": rel.name, "subfolder": str(rel.parent).replace("\\", "/"), "type": "output"}


def _image_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)


def _write_config(config_path: Path, config: dict[str, object]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_audio_input(path: Path, audio: object) -> None:
    if audio is None:
        raise ValueError("No audio input was provided")
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("Unsupported Comfy AUDIO object")

    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate", 44100))
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().cpu().float()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > waveform.shape[1]:
        # Most Comfy audio is channels x samples after batch removal. This keeps
        # obviously transposed mono/stereo tensors usable.
        waveform = waveform.T
    waveform = waveform.clamp(-1.0, 1.0)
    pcm = (waveform.T.numpy() * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(int(pcm.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _launch_process(config_path: Path, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    kwargs = {
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "cwd": str(_repo_root()),
        "text": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        [sys.executable, str(_runner_path()), "--config", str(config_path)],
        **kwargs,
    )


class LongformYvannLauncher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "launch_now": ("BOOLEAN", {"default": True}),
                "script_path": ("STRING", {"multiline": False, "default": "input/longform_script.txt"}),
                "audio_path": ("STRING", {"multiline": False, "default": "input/Temple_of_the_Scales.mp3"}),
                "global_style_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "abstract neon cyberpunk animation, glowing ribbon structures, tunnel geometry, cinematic volumetric light",
                    },
                ),
                "output_root": ("STRING", {"multiline": False, "default": "output/longform_yvann"}),
                "workflow_template_path": (
                    "STRING",
                    {"multiline": False, "default": "user/default/workflows/AudioReactive_ImagesToVideo_Yvann.json"},
                ),
                "chunk_duration_seconds": ("FLOAT", {"default": 45.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "overlap_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3599.0, "step": 1.0}),
                "render_profile": (["dj_final", "preview_fast", "custom"], {"default": "dj_final"}),
                "image_interval_seconds": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 600.0, "step": 1.0}),
                "render_width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
                "render_height": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
                "image_backend": ("STRING", {"multiline": False, "default": "comfy_api"}),
                "continuity_mode": ("STRING", {"multiline": False, "default": "style"}),
                "seed_strategy": ("STRING", {"multiline": False, "default": "derived"}),
                "base_seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1}),
                "resume": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "stop_on_failure": ("BOOLEAN", {"default": False}),
                "final_concat": ("BOOLEAN", {"default": True}),
                "yvann_render_fps": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "yvann_min_frames": ("INT", {"default": 24, "min": 8, "max": 1024, "step": 1}),
                "yvann_max_frames": ("INT", {"default": 720, "min": 8, "max": 2048, "step": 1}),
                "max_chunks": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                "uploaded_audio": ("AUDIO", {}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("job_id", "job_dir", "config_path")
    FUNCTION = "launch_job"
    OUTPUT_NODE = True
    CATEGORY = "Yvann/Longform"

    def launch_job(
        self,
        launch_now,
        script_path,
        audio_path,
        global_style_prompt,
        output_root,
        workflow_template_path,
        chunk_duration_seconds,
        overlap_seconds,
        render_profile,
        image_interval_seconds,
        render_width,
        render_height,
        image_backend,
        continuity_mode,
        seed_strategy,
        base_seed,
        resume,
        overwrite,
        stop_on_failure,
        final_concat,
        yvann_render_fps,
        yvann_min_frames,
        yvann_max_frames,
        max_chunks,
    ):
        if not bool(launch_now):
            return ("launch_disabled", "", "")

        repo_root = _repo_root()
        resolved_output_root = _resolve_path(output_root)
        resolved_output_root.mkdir(parents=True, exist_ok=True)

        job_id = _job_id()
        job_dir = resolved_output_root / job_id
        config = {
            "job_id": job_id,
            "script_path": str(_resolve_path(script_path)),
            "audio_path": str(_resolve_path(audio_path)),
            "global_style_prompt": global_style_prompt,
            "output_root": str(resolved_output_root),
            "comfy_api_url": "http://127.0.0.1:18188",
            "comfy_api_verify_tls": False,
            "workflow_template_path": str(_resolve_path(workflow_template_path)),
            "comfy_root": str(repo_root),
            "chunk_duration_seconds": float(chunk_duration_seconds),
            "overlap_seconds": float(overlap_seconds),
            "segmentation_mode": "auto",
            "motifs": [],
            "negative_prompt": "low quality, blurry, watermark, text artifacts",
            "continuity_mode": continuity_mode,
            "render_profile": render_profile,
            "image_backend": image_backend,
            "image_interval_seconds": float(image_interval_seconds),
            "image_width": int(render_width),
            "image_height": int(render_height),
            "comfy_t2i_checkpoint": "DreamShaper_8_pruned.safetensors",
            "comfy_t2i_steps": 10,
            "comfy_t2i_cfg": 4.5,
            "comfy_t2i_sampler": "euler",
            "comfy_t2i_scheduler": "normal",
            "seed_strategy": seed_strategy,
            "base_seed": int(base_seed),
            "seed_offset": 1009,
            "resume": bool(resume),
            "overwrite": bool(overwrite),
            "stop_on_failure": bool(stop_on_failure),
            "final_concat": bool(final_concat),
            "ffmpeg_video_codec": "libx264",
            "ffmpeg_crf": 22,
            "yvann_output_node_title": "First Pass | Low Res",
            "yvann_render_fps": float(yvann_render_fps),
            "yvann_min_frames": int(yvann_min_frames),
            "yvann_max_frames": int(yvann_max_frames),
            "max_chunks": int(max_chunks) if int(max_chunks) > 0 else None,
        }

        config_path = _config_path(resolved_output_root, job_id)
        log_path = job_dir / "launcher.log"
        _write_config(config_path, config)
        proc = _launch_process(config_path, log_path)

        JOB_REGISTRY[job_id] = {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "pid": proc.pid,
            "started_at": time.time(),
            "status": "running",
        }

        return (job_id, str(job_dir), str(config_path))


class LongformYvannCueSheetLauncher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "launch_now": ("BOOLEAN", {"default": True}),
                "cue_sheet_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "00:00:00  1 Deep Hertz - Melting Sun  # A. 00:00:00 Rocket preparing for launch. "
                            "Close-ups of the rocket, smoke and ice falling.\n"
                            "00:04:39  2 Miguel Montero - Captain Hook  # B. 00:03:30 Rocket taking off, "
                            "climbing, stage separation."
                        ),
                    },
                ),
                "audio_path": ("STRING", {"multiline": False, "default": "input/Temple_of_the_Scales.mp3"}),
                "global_style_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "cinematic audio-reactive visuals, high detail, coherent motion, immersive lighting",
                    },
                ),
                "output_root": ("STRING", {"multiline": False, "default": "output/longform_yvann"}),
                "yvann_workflow_path": (
                    "STRING",
                    {"multiline": False, "default": "user/default/workflows/AudioReactive_ImagesToVideo_Yvann.json"},
                ),
                "chunk_duration_seconds": ("FLOAT", {"default": 45.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "render_profile": (["dj_final", "preview_fast", "custom"], {"default": "dj_final"}),
                "image_interval_seconds": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 600.0, "step": 1.0}),
                "render_width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
                "render_height": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
                "image_backend": ("STRING", {"multiline": False, "default": "comfy_api"}),
                "base_seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1}),
                "resume": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "final_concat": ("BOOLEAN", {"default": True}),
                "yvann_render_fps": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "yvann_max_frames": ("INT", {"default": 720, "min": 8, "max": 2048, "step": 1}),
                "max_chunks": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "uploaded_audio": ("AUDIO", {}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("job_id", "job_dir", "config_path")
    FUNCTION = "launch_job"
    OUTPUT_NODE = True
    CATEGORY = "Yvann/Longform"

    def launch_job(
        self,
        launch_now,
        cue_sheet_text,
        audio_path,
        global_style_prompt,
        output_root,
        yvann_workflow_path,
        chunk_duration_seconds,
        render_profile,
        image_interval_seconds,
        render_width,
        render_height,
        image_backend,
        base_seed,
        resume,
        overwrite,
        final_concat,
        yvann_render_fps,
        yvann_max_frames,
        max_chunks,
        uploaded_audio=None,
    ):
        if not bool(launch_now):
            return ("launch_disabled", "", "")

        if not str(cue_sheet_text).strip():
            raise ValueError("cue_sheet_text is empty")

        repo_root = _repo_root()
        resolved_output_root = _resolve_path(output_root)
        resolved_output_root.mkdir(parents=True, exist_ok=True)

        job_id = _job_id()
        job_dir = resolved_output_root / job_id
        config_path = _config_path(resolved_output_root, job_id)
        cue_sheet_path = config_path.with_suffix(".cuesheet.txt")
        _write_text(cue_sheet_path, str(cue_sheet_text).strip() + "\n")
        if uploaded_audio is not None:
            audio_source_path = config_path.with_suffix(".uploaded_audio.wav")
            _write_audio_input(audio_source_path, uploaded_audio)
        else:
            audio_source_path = _resolve_path(audio_path)

        config = {
            "job_id": job_id,
            "script_path": str(cue_sheet_path),
            "audio_path": str(audio_source_path),
            "global_style_prompt": global_style_prompt,
            "output_root": str(resolved_output_root),
            "comfy_api_url": "http://127.0.0.1:18188",
            "comfy_api_verify_tls": False,
            "workflow_template_path": str(_resolve_path(yvann_workflow_path)),
            "comfy_root": str(repo_root),
            "chunk_duration_seconds": float(chunk_duration_seconds),
            "overlap_seconds": 0.0,
            "segmentation_mode": "auto",
            "motifs": [],
            "negative_prompt": "low quality, blurry, watermark, text artifacts",
            "continuity_mode": "style",
            "render_profile": render_profile,
            "image_backend": image_backend,
            "image_interval_seconds": float(image_interval_seconds),
            "image_width": int(render_width),
            "image_height": int(render_height),
            "comfy_t2i_checkpoint": "DreamShaper_8_pruned.safetensors",
            "comfy_t2i_steps": 10,
            "comfy_t2i_cfg": 4.5,
            "comfy_t2i_sampler": "euler",
            "comfy_t2i_scheduler": "normal",
            "seed_strategy": "derived",
            "base_seed": int(base_seed),
            "seed_offset": 1009,
            "resume": bool(resume),
            "overwrite": bool(overwrite),
            "stop_on_failure": False,
            "final_concat": bool(final_concat),
            "ffmpeg_video_codec": "libx264",
            "ffmpeg_crf": 22,
            "yvann_output_node_title": "First Pass | Low Res",
            "yvann_render_fps": float(yvann_render_fps),
            "yvann_min_frames": 24,
            "yvann_max_frames": int(yvann_max_frames),
            "max_chunks": int(max_chunks) if int(max_chunks) > 0 else None,
        }

        log_path = job_dir / "launcher.log"
        _write_config(config_path, config)
        proc = _launch_process(config_path, log_path)

        JOB_REGISTRY[job_id] = {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "config_path": str(config_path),
            "cue_sheet_path": str(cue_sheet_path),
            "log_path": str(log_path),
            "pid": proc.pid,
            "started_at": time.time(),
            "status": "running",
        }

        return (job_id, str(job_dir), str(config_path))


class LongformYvannCueSheetParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cue_sheet_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "00:00:00  1 Deep Hertz - Melting Sun  # A. 00:00:00 Rocket preparing for launch. "
                            "Close-ups of the rocket, smoke and ice falling.\n"
                            "00:04:39  2 Miguel Montero - Captain Hook  # B. 00:03:30 Rocket taking off, "
                            "climbing, stage separation."
                        ),
                    },
                ),
                "audio_path": ("STRING", {"multiline": False, "default": "input/Temple_of_the_Scales.mp3"}),
                "chunk_duration_seconds": ("FLOAT", {"default": 45.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "max_chunks": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("cue_sheet_text", "cue_json", "batch_plan")
    FUNCTION = "parse"
    OUTPUT_NODE = True
    CATEGORY = "Yvann/Longform"

    def parse(self, cue_sheet_text, audio_path, chunk_duration_seconds, max_chunks):
        resolved_audio = _resolve_path(audio_path)
        duration = _audio_duration(resolved_audio)
        cues = _extract_cues(str(cue_sheet_text), duration)
        if not cues:
            message = "No visual cue markers found. Add comments like: # A. 00:00:00 Rocket preparing for launch"
            return {"ui": {"text": [message]}, "result": (str(cue_sheet_text), "[]", message)}

        total_duration = duration or float(cues[-1]["end"])
        chunks = _plan_chunks(
            total_duration,
            [float(cue["start"]) for cue in cues],
            float(chunk_duration_seconds),
            int(max_chunks),
        )
        chunk_entries = []
        for idx, (start, end) in enumerate(chunks, start=1):
            cue = next((c for c in cues if float(c["start"]) <= ((start + end) * 0.5) < float(c["end"])), cues[-1])
            chunk_entries.append(
                {
                    "chunk_id": f"chunk_{idx:04d}",
                    "start": start,
                    "end": end,
                    "visual_batch_id": cue["id"],
                    "summary": cue["summary"],
                }
            )

        parsed = {
            "audio_path": str(resolved_audio),
            "audio_duration": total_duration,
            "visual_batches": cues,
            "chunks": chunk_entries,
        }
        lines = [
            "Parsed cue sheet",
            f"Audio: {resolved_audio}",
            f"Detected audio duration: {_sec_to_hms(total_duration) if duration else 'unknown; using cue ranges'}",
            f"Visual batches: {len(cues)}",
            f"Render chunks: {len(chunks)}",
            "",
            "Visual batches:",
        ]
        for cue in cues:
            lines.append(
                f"{cue['id']}  {_sec_to_hms(float(cue['start']))}-{_sec_to_hms(float(cue['end']))}  {str(cue['summary'])[:180]}"
            )
        lines.append("")
        lines.append("Render chunks:")
        for chunk in chunk_entries:
            lines.append(
                f"{chunk['chunk_id']}  {_sec_to_hms(float(chunk['start']))}-{_sec_to_hms(float(chunk['end']))}  batch {chunk['visual_batch_id']}"
            )
        batch_plan = "\n".join(lines)
        return {"ui": {"text": lines}, "result": (str(cue_sheet_text), json.dumps(parsed, indent=2), batch_plan)}


class LongformYvannCueSheetBatchPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cue_sheet_text": ("STRING", {"multiline": True, "default": ""}),
                "audio_path": ("STRING", {"multiline": False, "default": "input/Temple_of_the_Scales.mp3"}),
                "chunk_duration_seconds": ("FLOAT", {"default": 45.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "max_chunks": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("batch_plan",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Yvann/Longform"

    def preview(self, cue_sheet_text, audio_path, chunk_duration_seconds, max_chunks):
        resolved_audio = _resolve_path(audio_path)
        duration = _audio_duration(resolved_audio)
        cues = _extract_cues(str(cue_sheet_text), duration)
        if not cues:
            message = "No visual cue markers found. Add comments like: # A. 00:00:00 Rocket preparing for launch"
            return {"ui": {"text": [message]}, "result": (message,)}

        total_duration = duration or float(cues[-1]["end"])
        chunks = _plan_chunks(
            total_duration,
            [float(cue["start"]) for cue in cues],
            float(chunk_duration_seconds),
            int(max_chunks),
        )

        lines = [
            "Cue-sheet batch plan",
            f"Audio: {resolved_audio}",
            f"Detected audio duration: {_sec_to_hms(total_duration) if duration else 'unknown; using cue ranges'}",
            f"Visual batches: {len(cues)}",
            f"Render chunks shown: {len(chunks)}",
            "",
            "Visual batches:",
        ]
        for cue in cues:
            lines.append(
                f"{cue['id']}  {_sec_to_hms(float(cue['start']))}-{_sec_to_hms(float(cue['end']))}  {str(cue['summary'])[:180]}"
            )
        lines.append("")
        lines.append("Render chunks:")
        for idx, (start, end) in enumerate(chunks, start=1):
            cue = next((c for c in cues if float(c["start"]) <= ((start + end) * 0.5) < float(c["end"])), cues[-1])
            lines.append(f"chunk_{idx:04d}  {_sec_to_hms(start)}-{_sec_to_hms(end)}  batch {cue['id']}")

        text = "\n".join(lines)
        return {"ui": {"text": lines}, "result": (text,)}


class LongformYvannJobStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_id": ("STRING", {"multiline": False, "default": ""}),
                "job_dir": ("STRING", {"multiline": False, "default": ""}),
                "config_path": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "output_root": ("STRING", {"multiline": False, "default": "output/longform_yvann"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "show_status"
    OUTPUT_NODE = True
    CATEGORY = "Yvann/Longform"

    def show_status(self, job_id, job_dir, config_path, output_root="output/longform_yvann"):
        resolved_output_root = _resolve_path(str(output_root))
        resolved_job_dir = Path(str(job_dir).strip()) if str(job_dir).strip() else None
        if resolved_job_dir and not resolved_job_dir.is_absolute():
            resolved_job_dir = (_repo_root() / resolved_job_dir).resolve()
        if str(job_id).strip() == "launch_disabled":
            resolved_job_dir = _latest_manifest_job_dir(resolved_output_root)
        if not resolved_job_dir or not resolved_job_dir.exists():
            resolved_job_dir = _latest_manifest_job_dir(resolved_output_root)
        if resolved_job_dir and not (resolved_job_dir / "manifest" / "chunk_manifest.json").exists():
            resolved_job_dir = _latest_manifest_job_dir(resolved_output_root)

        lines = [
            f"job_id: {job_id}",
            f"job_dir: {resolved_job_dir or job_dir}",
            f"config_path: {config_path}",
            "Outputs appear inside the job folder as images/, videos/, and final/final_concat.mp4.",
        ]
        ui_images: list[dict[str, str]] = []

        if not resolved_job_dir or not resolved_job_dir.exists():
            lines.append("No job folder found yet. Queue with launch_now=true to start a job.")
            return {"ui": {"text": lines}}

        state_path = resolved_job_dir / "job_state.json"
        manifest_path = resolved_job_dir / "manifest" / "chunk_manifest.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                lines.append(
                    f"progress: {len(state.get('completed_chunks', []))}/{state.get('number_of_chunks', '?')} chunks completed; "
                    f"failed: {len(state.get('failed_chunks', []))}; concat: {state.get('final_concat_status')}"
                )
            except Exception as exc:
                lines.append(f"Could not read job_state.json: {exc}")

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                lines.append("")
                lines.append("Batches/chunks and generated images:")
                for chunk in manifest.get("chunks", [])[:80]:
                    image_paths = [Path(p) for p in chunk.get("assigned_image_paths", [])]
                    for image_path in image_paths:
                        entry = _image_ui_entry(image_path)
                        if entry and len(ui_images) < 64:
                            ui_images.append(entry)
                    image_names = ", ".join(path.name for path in image_paths) or "images pending"
                    lines.append(
                        f"{chunk.get('chunk_id')} {_sec_to_hms(float(chunk.get('start_time', 0)))}-"
                        f"{_sec_to_hms(float(chunk.get('end_time', 0)))} batch {chunk.get('visual_batch_id') or '?'} "
                        f"{chunk.get('status')}: {image_names}"
                    )
            except Exception as exc:
                lines.append(f"Could not read chunk_manifest.json: {exc}")
        else:
            lines.append("Manifest not written yet. Queue this status node again in a moment to refresh.")

        ui: dict[str, object] = {"text": lines}
        if ui_images:
            ui["images"] = ui_images
        return {"ui": ui}


class LongformYvannGeneratedImagesOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_dir": ("STRING", {"multiline": False, "default": ""}),
                "output_root": ("STRING", {"multiline": False, "default": "output/longform_yvann"}),
                "batch_or_chunk_filter": ("STRING", {"multiline": False, "default": ""}),
                "max_images": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "image_manifest")
    FUNCTION = "load_images"
    OUTPUT_NODE = True
    CATEGORY = "Yvann/Longform"

    def load_images(self, job_dir, output_root, batch_or_chunk_filter, max_images):
        resolved_job_dir = Path(str(job_dir).strip()) if str(job_dir).strip() else None
        if resolved_job_dir and not resolved_job_dir.is_absolute():
            resolved_job_dir = (_repo_root() / resolved_job_dir).resolve()
        if not resolved_job_dir or not resolved_job_dir.exists():
            resolved_job_dir = _latest_manifest_job_dir(_resolve_path(str(output_root)))

        lines: list[str] = []
        ui_images: list[dict[str, str]] = []
        tensors: list[torch.Tensor] = []
        if not resolved_job_dir or not resolved_job_dir.exists():
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            text = "No generated image job found yet. Queue the generator first."
            return {"ui": {"text": [text]}, "result": (placeholder, text)}

        manifest_path = resolved_job_dir / "manifest" / "chunk_manifest.json"
        if not manifest_path.exists():
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            text = f"No chunk manifest found in {resolved_job_dir}. Images are not ready yet."
            return {"ui": {"text": [text]}, "result": (placeholder, text)}

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        filter_value = str(batch_or_chunk_filter).strip().lower()
        for chunk in manifest.get("chunks", []):
            batch_id = str(chunk.get("visual_batch_id") or "").lower()
            chunk_id = str(chunk.get("chunk_id") or "").lower()
            if filter_value and filter_value not in batch_id and filter_value not in chunk_id:
                continue
            for image_path_text in chunk.get("assigned_image_paths", []):
                if len(tensors) >= int(max_images):
                    break
                image_path = Path(image_path_text)
                if not image_path.exists():
                    continue
                tensors.append(_image_to_tensor(image_path))
                entry = _image_ui_entry(image_path)
                if entry:
                    ui_images.append(entry)
                lines.append(
                    f"{chunk.get('chunk_id')} {_sec_to_hms(float(chunk.get('start_time', 0)))}-"
                    f"{_sec_to_hms(float(chunk.get('end_time', 0)))} batch {chunk.get('visual_batch_id')}: {image_path.name}"
                )
            if len(tensors) >= int(max_images):
                break

        if not tensors:
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            text = f"No generated images matched filter '{batch_or_chunk_filter}' in {resolved_job_dir}."
            return {"ui": {"text": [text]}, "result": (placeholder, text)}

        # If image sizes differ, use Comfy-visible thumbnails in UI and return a stack resized by PIL.
        first_h, first_w = tensors[0].shape[0], tensors[0].shape[1]
        normalized: list[torch.Tensor] = []
        for tensor in tensors:
            if tensor.shape[0] == first_h and tensor.shape[1] == first_w:
                normalized.append(tensor)
                continue
            pil = Image.fromarray((tensor.numpy() * 255).astype(np.uint8)).resize((first_w, first_h), Image.Resampling.LANCZOS)
            normalized.append(torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0))

        batch = torch.stack(normalized, dim=0)
        text = "\n".join(lines)
        ui: dict[str, object] = {"text": lines}
        if ui_images:
            ui["images"] = ui_images
        return {"ui": ui, "result": (batch, text)}


class LongformYvannFourImagesOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_dir": ("STRING", {"multiline": False, "default": ""}),
                "output_root": ("STRING", {"multiline": False, "default": "output/longform_yvann"}),
                "batch_or_chunk_filter": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_scene_batch", "image_batch_manifest")
    FUNCTION = "load_scene_batch"
    OUTPUT_NODE = True
    CATEGORY = "Yvann/Longform"

    def load_scene_batch(self, job_dir, output_root, batch_or_chunk_filter):
        resolved_job_dir = Path(str(job_dir).strip()) if str(job_dir).strip() else None
        if resolved_job_dir and not resolved_job_dir.is_absolute():
            resolved_job_dir = (_repo_root() / resolved_job_dir).resolve()
        if not resolved_job_dir or not resolved_job_dir.exists():
            resolved_job_dir = _latest_manifest_job_dir(_resolve_path(str(output_root)))

        placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        if not resolved_job_dir or not resolved_job_dir.exists():
            text = "No generated image job found yet. Queue the generator first."
            return {"ui": {"text": [text]}, "result": (placeholder, text)}

        manifest_path = resolved_job_dir / "manifest" / "chunk_manifest.json"
        if not manifest_path.exists():
            text = f"No chunk manifest found in {resolved_job_dir}. Images are not ready yet."
            return {"ui": {"text": [text]}, "result": (placeholder, text)}

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        filter_value = str(batch_or_chunk_filter).strip().lower()
        selected_chunk = None
        for chunk in manifest.get("chunks", []):
            batch_id = str(chunk.get("visual_batch_id") or "").lower()
            chunk_id = str(chunk.get("chunk_id") or "").lower()
            if filter_value and filter_value not in batch_id and filter_value not in chunk_id:
                continue
            if chunk.get("assigned_image_paths"):
                selected_chunk = chunk
                break

        if selected_chunk is None:
            text = f"No generated images matched filter '{batch_or_chunk_filter}' in {resolved_job_dir}."
            return {"ui": {"text": [text]}, "result": (placeholder, text)}

        image_paths = [Path(p) for p in selected_chunk.get("assigned_image_paths", []) if Path(p).exists()]
        if not image_paths:
            text = f"Selected chunk has no generated image files yet: {selected_chunk.get('chunk_id')}"
            return {"ui": {"text": [text]}, "result": (placeholder, text)}

        tensors: list[torch.Tensor] = []
        ui_images: list[dict[str, str]] = []
        lines = [
            f"Selected chunk: {selected_chunk.get('chunk_id')} batch {selected_chunk.get('visual_batch_id')}",
            f"Time: {_sec_to_hms(float(selected_chunk.get('start_time', 0)))}-{_sec_to_hms(float(selected_chunk.get('end_time', 0)))}",
            f"Generated scene batch images: {len(image_paths)}",
        ]
        for image_path in image_paths:
            tensor = _image_to_tensor(image_path)
            tensors.append(tensor)
            entry = _image_ui_entry(image_path)
            if entry:
                ui_images.append(entry)
            lines.append(image_path.name)

        first_h, first_w = tensors[0].shape[0], tensors[0].shape[1]
        normalized: list[torch.Tensor] = []
        for tensor in tensors:
            if tensor.shape[0] == first_h and tensor.shape[1] == first_w:
                normalized.append(tensor)
                continue
            pil = Image.fromarray((tensor.numpy() * 255).astype(np.uint8)).resize((first_w, first_h), Image.Resampling.LANCZOS)
            normalized.append(torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0))

        ui: dict[str, object] = {"text": lines}
        if ui_images:
            ui["images"] = ui_images
        text = "\n".join(lines)
        return {"ui": ui, "result": (torch.stack(normalized, dim=0), text)}


class LongformYvannWorkflowInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yvann_workflow_path": (
                    "STRING",
                    {"multiline": False, "default": "user/default/workflows/AudioReactive_ImagesToVideo_Yvann.json"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("yvann_node_report",)
    FUNCTION = "inspect"
    OUTPUT_NODE = True
    CATEGORY = "Yvann/Longform"

    def inspect(self, yvann_workflow_path):
        workflow_path = _resolve_path(str(yvann_workflow_path))
        if not workflow_path.exists():
            text = f"Yvann workflow not found: {workflow_path}"
            return {"ui": {"text": [text]}, "result": (text,)}

        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        load_images = []
        batch_loaders = []
        load_audio = []
        audio_nodes = []
        outputs = []
        for node in workflow.get("nodes", []):
            node_type = str(node.get("type", ""))
            title = node.get("title") or node.get("type") or ""
            title_lower = str(title).lower()
            node_id = node.get("id")
            if node_type == "LoadImage":
                load_images.append(f"{node_id} {title}")
            elif node_type in {"LoadImagesFromFolderKJ", "VHS_LoadImagesPath"}:
                batch_loaders.append(f"{node_id} {node_type} {title}")
            elif node_type == "LoadAudio" or title_lower == "load audio":
                load_audio.append(f"{node_id} {title}")
            elif "Audio" in node_type or "audio" in title_lower:
                audio_nodes.append(f"{node_id} {node_type} {title}")
            elif node_type == "VHS_VideoCombine":
                outputs.append(f"{node_id} {title}")

        lines = [
            "Actual Yvann render engine",
            f"Workflow file: {workflow_path}",
            "",
            "The longform launcher does not render video itself. For every timestamp chunk it:",
            "1. generates a full image folder for the timestamp batch,",
            "2. points the Yvann folder batch loader at that image folder,",
            "3. replaces the Yvann LoadAudio node with the chunk audio,",
            "4. queues the Yvann video output node.",
            "",
            "Yvann folder batch loaders:",
        ]
        lines.extend([f"- {item}" for item in batch_loaders] or ["- none found"])
        if load_images:
            lines.append("")
            lines.append("Legacy LoadImage nodes still present but bypassed/collapsed:")
            lines.extend([f"- {item}" for item in load_images])
        lines.append("")
        lines.append("Yvann audio input replaced per chunk:")
        lines.extend([f"- {item}" for item in load_audio] or ["- none found"])
        lines.append("")
        lines.append("Yvann audio-reactive processing nodes:")
        lines.extend([f"- {item}" for item in audio_nodes[:12]] or ["- none found"])
        lines.append("")
        lines.append("Yvann video outputs available:")
        lines.extend([f"- {item}" for item in outputs] or ["- none found"])

        text = "\n".join(lines)
        return {"ui": {"text": lines}, "result": (text,)}


@PromptServer.instance.routes.get("/yvann_longform/jobs")
async def get_jobs(_request):
    jobs = []
    for job_id, info in JOB_REGISTRY.items():
        pid = int(info.get("pid", 0))
        running = False
        try:
            os.kill(pid, 0)
            running = True
        except Exception:
            running = False
        jobs.append({**info, "running": running})
    return web.json_response({"jobs": jobs})


NODE_CLASS_MAPPINGS = {
    "LongformYvannLauncher": LongformYvannLauncher,
    "LongformYvannCueSheetParser": LongformYvannCueSheetParser,
    "LongformYvannCueSheetLauncher": LongformYvannCueSheetLauncher,
    "LongformYvannCueSheetBatchPlan": LongformYvannCueSheetBatchPlan,
    "LongformYvannJobStatus": LongformYvannJobStatus,
    "LongformYvannGeneratedImagesOutput": LongformYvannGeneratedImagesOutput,
    "LongformYvannFourImagesOutput": LongformYvannFourImagesOutput,
    "LongformYvannWorkflowInspector": LongformYvannWorkflowInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongformYvannLauncher": "Yvann Longform Launcher",
    "LongformYvannCueSheetParser": "Yvann Cue Sheet Parser",
    "LongformYvannCueSheetLauncher": "Yvann Cue Sheet Batch Generator",
    "LongformYvannCueSheetBatchPlan": "Yvann Cue Sheet Batch Plan",
    "LongformYvannJobStatus": "Yvann Longform Job Status",
    "LongformYvannGeneratedImagesOutput": "Yvann Generated Batch Images Output",
    "LongformYvannFourImagesOutput": "Yvann Generated Scene Batch Output",
    "LongformYvannWorkflowInspector": "Yvann Render Engine Inspector",
}
