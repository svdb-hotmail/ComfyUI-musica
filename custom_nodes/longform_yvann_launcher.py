from __future__ import annotations

import json
import os
import wave
import subprocess
import sys
import time
from pathlib import Path
import re
from typing import Any

from aiohttp import web
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw

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


def _process_record_path(config_path: Path) -> Path:
    return config_path.with_suffix(".process.json")


def _cancel_path(job_dir: Path) -> Path:
    return job_dir / "cancel.requested"


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_config(value: object, default: dict[str, object] | None = None) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return dict(default or {})
    text = str(value).strip()
    if not text:
        return dict(default or {})
    try:
        loaded = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid JSON config input: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("JSON config input must be an object")
    return loaded


def _audio_file_choices() -> list[str]:
    roots = [_repo_root() / "input", _repo_root() / "output" / "longform_yvann" / "_uploaded_audio"]
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    choices = [""]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                try:
                    choices.append(str(path.resolve().relative_to(_repo_root().resolve())).replace("\\", "/"))
                except Exception:
                    choices.append(str(path))
    fallback = "input/Temple_of_the_Scales.mp3"
    if fallback not in choices:
        choices.append(fallback)
    return choices


def _pid_running(pid: object) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
        return True
    except Exception:
        return False


def _job_dir_from_config(config: dict[str, object]) -> Path:
    return Path(str(config["output_root"])).resolve() / str(config["job_id"])


def _state_summary(job_dir: Path) -> dict[str, object]:
    state_path = job_dir / "job_state.json"
    if not state_path.exists():
        return {"status": "starting", "completed": 0, "failed": 0, "chunks": "?"}
    try:
        state = _read_json(state_path)
    except Exception as exc:
        return {"status": f"state_unreadable: {exc}", "completed": 0, "failed": 0, "chunks": "?"}
    completed = len(state.get("completed_chunks", []))
    failed = len(state.get("failed_chunks", []))
    chunks = state.get("number_of_chunks", "?")
    status = state.get("status")
    if not status:
        if state.get("final_concat_status") == "completed" and completed == chunks:
            status = "completed"
        elif failed:
            status = "failed"
        else:
            status = "running"
    return {
        "status": status,
        "completed": completed,
        "failed": failed,
        "chunks": chunks,
        "current_chunk_index": state.get("current_chunk_index", 0),
        "final_concat_status": state.get("final_concat_status"),
        "updated_at": state.get("updated_at"),
        "cancel_requested": bool(state.get("cancel_requested")) or _cancel_path(job_dir).exists(),
    }


def _discover_jobs(output_root: Path | None = None) -> list[dict[str, object]]:
    search_roots = [output_root.resolve()] if output_root else [(_repo_root() / "output").resolve()]
    discovered: dict[str, dict[str, object]] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for config_path in root.rglob("_launcher_configs/*.json"):
            try:
                config = _read_json(config_path)
                job_id = str(config.get("job_id") or config_path.stem)
                job_dir = _job_dir_from_config(config)
                process_path = _process_record_path(config_path)
                process = _read_json(process_path) if process_path.exists() else {}
                summary = _state_summary(job_dir)
                discovered[job_id] = {
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "config_path": str(config_path),
                    "process_record_path": str(process_path),
                    "pid": process.get("pid"),
                    "running": _pid_running(process.get("pid")),
                    "started_at": process.get("started_at"),
                    **summary,
                }
            except Exception as exc:
                discovered[str(config_path)] = {"job_id": config_path.stem, "config_path": str(config_path), "status": f"unreadable: {exc}", "running": False}

        for state_path in root.rglob("job_state.json"):
            job_dir = state_path.parent
            job_id = job_dir.name
            if job_id in discovered:
                continue
            summary = _state_summary(job_dir)
            discovered[job_id] = {
                "job_id": job_id,
                "job_dir": str(job_dir),
                "config_path": str(job_dir / "job_config.json"),
                "process_record_path": "",
                "pid": None,
                "running": False,
                **summary,
            }

    for job_id, info in JOB_REGISTRY.items():
        job_dir = Path(str(info.get("job_dir", "")))
        summary = _state_summary(job_dir) if job_dir.exists() else {}
        discovered[job_id] = {**info, **summary, "running": _pid_running(info.get("pid"))}
    return sorted(discovered.values(), key=lambda item: str(item.get("started_at") or item.get("job_id")), reverse=True)


def _find_job(job_id: str, output_root: Path | None = None) -> dict[str, object] | None:
    jobs = _discover_jobs(output_root)
    if not str(job_id).strip():
        return jobs[0] if jobs else None
    for job in jobs:
        if str(job.get("job_id")) == str(job_id):
            return job
    return None


def _request_cancel(job_id: str, job_dir: Path) -> dict[str, object]:
    job_dir.mkdir(parents=True, exist_ok=True)
    cancel_path = _cancel_path(job_dir)
    cancel_path.write_text(f"cancel_requested_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n", encoding="utf-8")
    state_path = job_dir / "job_state.json"
    if state_path.exists():
        try:
            state = _read_json(state_path)
            state["cancel_requested"] = True
            if state.get("status") not in {"completed", "failed", "cancelled"}:
                state["status"] = "cancelling"
            state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            timestamps = state.setdefault("timestamps", {})
            if isinstance(timestamps, dict):
                timestamps["last_update"] = state["updated_at"]
            _write_json(state_path, state)
        except Exception:
            pass
    registry_info = JOB_REGISTRY.get(job_id)
    if registry_info is not None:
        registry_info["status"] = "cancelling"
    return {"job_id": job_id, "job_dir": str(job_dir), "cancel_path": str(cancel_path), "status": "cancelling"}


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


def _audio_files() -> list[str]:
    input_root = _repo_root() / "input"
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    fallback = "input/Temple_of_the_Scales.mp3"
    files = [""]
    if not input_root.exists():
        return files
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            try:
                files.append("input/" + str(path.relative_to(input_root)).replace("\\", "/"))
            except Exception:
                files.append(str(path))
    if len(files) == 1 and (_repo_root() / fallback).exists():
        files.append(fallback)
    return files


def _decode_json_object(value: object, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return payload


def _write_audio_waveform_preview(audio_path: Path, output_path: Path, width: int = 1280, height: int = 240) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "12000",
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    image = Image.new("RGB", (width, height), (18, 22, 30))
    draw = ImageDraw.Draw(image)
    center = height // 2
    draw.line((0, center, width, center), fill=(64, 78, 96), width=1)
    if samples.size:
        samples /= max(1.0, float(np.max(np.abs(samples))))
        bucket = max(1, int(np.ceil(samples.size / width)))
        for x in range(width):
            segment = samples[x * bucket : min(samples.size, (x + 1) * bucket)]
            if segment.size == 0:
                continue
            lo = float(segment.min())
            hi = float(segment.max())
            y1 = int(center - hi * (height * 0.44))
            y2 = int(center - lo * (height * 0.44))
            draw.line((x, y1, x, y2), fill=(76, 194, 178), width=1)
    image.save(output_path)
    return output_path


def _normalize_render_profile(value: object) -> str:
    profile = str(value or "balanced").strip().lower()
    if profile in {"draft", "preview_fast", "preview", "fast"}:
        return "draft"
    if profile in {"balanced", "default"}:
        return "balanced"
    if profile in {"dj_final", "final", "production"}:
        return "final"
    if profile == "custom":
        return "custom"
    if profile.isdigit():
        return "balanced"
    raise ValueError("render_profile must be 'draft', 'balanced', 'final', or 'custom'")


def _profile_override_note(render_profile: object) -> str:
    profile = _normalize_render_profile(render_profile)
    if profile == "custom":
        return "custom profile: manual custom-only controls are active"
    return f"{profile} profile: manual custom-only controls were ignored by the backend profile preset"


class LongformYvannRenderProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "render_profile": (["draft", "balanced", "final", "custom"], {"default": "balanced", "tooltip": "Preset profiles own the custom override values. Use custom to apply the manual fields below."}),
                "custom_image_interval_seconds": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 600.0, "step": 1.0}),
                "custom_render_width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
                "custom_render_height": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
                "custom_t2i_steps": ("INT", {"default": 10, "min": 1, "max": 150, "step": 1}),
                "custom_t2i_cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "custom_yvann_render_fps": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "custom_yvann_min_frames": ("INT", {"default": 24, "min": 8, "max": 2048, "step": 1}),
                "custom_yvann_max_frames": ("INT", {"default": 192, "min": 8, "max": 4096, "step": 1}),
                "custom_ffmpeg_crf": ("INT", {"default": 22, "min": 0, "max": 51, "step": 1}),
            }
        }

    RETURN_TYPES = ("YVANN_RENDER_PROFILE", "STRING")
    RETURN_NAMES = ("profile_config", "profile_summary")
    FUNCTION = "build"
    CATEGORY = "Yvann/Longform"

    @classmethod
    def VALIDATE_INPUTS(cls, render_profile=None):
        _normalize_render_profile(render_profile)
        return True

    def build(self, render_profile, custom_image_interval_seconds, custom_render_width, custom_render_height, custom_t2i_steps, custom_t2i_cfg, custom_yvann_render_fps, custom_yvann_min_frames, custom_yvann_max_frames, custom_ffmpeg_crf):
        profile = _normalize_render_profile(render_profile)
        payload = {
            "render_profile": profile,
            "profile_behavior": _profile_override_note(profile),
            "image_interval_seconds": float(custom_image_interval_seconds),
            "image_width": int(custom_render_width),
            "image_height": int(custom_render_height),
            "comfy_t2i_steps": int(custom_t2i_steps),
            "comfy_t2i_cfg": float(custom_t2i_cfg),
            "yvann_render_fps": float(custom_yvann_render_fps),
            "yvann_min_frames": int(custom_yvann_min_frames),
            "yvann_max_frames": int(custom_yvann_max_frames),
            "ffmpeg_crf": int(custom_ffmpeg_crf),
        }
        summary = f"{profile}: {_profile_override_note(profile)}"
        return (json.dumps(payload, ensure_ascii=True), summary)


class LongformYvannCueSheetSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cue_sheet_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "Single source of truth for the longform cue sheet. Link this into both launcher and preview nodes.",
                        "default": (
                            "00:00:00  1 Track / section name  # A. 00:00:00 Describe the first visual scene here.\n"
                            "00:00:45  2 Next section          # B. 00:00:45 Describe the next visual scene here."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cue_sheet_text",)
    FUNCTION = "emit"
    CATEGORY = "Yvann/Longform"

    def emit(self, cue_sheet_text):
        text = str(cue_sheet_text).strip()
        if not text:
            raise ValueError("cue_sheet_text is empty")
        return (text,)


class LongformYvannAudioSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_mode": (["uploaded_audio", "input_file", "path"], {"default": "input_file", "tooltip": "Use uploaded_audio when an AUDIO input is connected, pick an existing input file, or type a path."}),
                "input_audio_file": (_audio_files(), {"default": "", "tooltip": "Existing audio under the ComfyUI input folder."}),
                "audio_path": ("STRING", {"multiline": False, "default": "", "tooltip": "Used when source_mode is path. Relative paths resolve from the ComfyUI root."}),
            },
            "optional": {
                "uploaded_audio": ("AUDIO", {"tooltip": "Connect a LoadAudio/UploadAudio style node here. The audio is written to input/yvann_uploads for backend use."}),
            },
        }

    RETURN_TYPES = ("YVANN_AUDIO_SOURCE", "STRING")
    RETURN_NAMES = ("audio_source", "audio_path")
    FUNCTION = "resolve"
    CATEGORY = "Yvann/Longform"

    def resolve(self, source_mode, input_audio_file, audio_path, uploaded_audio=None):
        mode = str(source_mode or "input_file")
        if uploaded_audio is not None and mode == "uploaded_audio":
            target = _repo_root() / "input" / "yvann_uploads" / f"uploaded_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            _write_audio_input(target, uploaded_audio)
            resolved = target
        elif mode == "path":
            if not str(audio_path).strip():
                raise ValueError("audio_path is empty. Type an audio path or switch source_mode to input_file/uploaded_audio.")
            resolved = _resolve_path(str(audio_path))
        else:
            selected = str(input_audio_file).strip()
            if not selected:
                raise ValueError("No existing audio file selected. Pick an input_audio_file or switch source_mode to uploaded_audio/path.")
            resolved = _resolve_path(selected)
        payload = {"audio_path": str(resolved), "source_mode": mode}
        return (json.dumps(payload, ensure_ascii=True), str(resolved))


class LongformYvannExecutionSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yvann_workflow_path": ("STRING", {"multiline": False, "default": "custom_nodes/comfyui_yvann-nodes/example_workflows/AudioReactive_ImagesToVideo_Yvann.json"}),
                "yvann_output_node_title": ("STRING", {"multiline": False, "default": "First Pass | Low Res", "tooltip": "Preferred VHS_VideoCombine title in the hidden Yvann workflow."}),
                "yvann_audio_analysis_mode": (["Full Audio", "Drums Only", "Vocals Only", "Bass Only", "Others Audio"], {"default": "Full Audio", "tooltip": "Audio Analysis mode patched into Yvann Audio Analysis nodes when present."}),
                "image_backend": (["comfy_api", "procedural"], {"default": "comfy_api"}),
                "continuity_mode": (["style", "carry", "independent"], {"default": "style", "tooltip": "style reuses prompt language; carry also uses previous keyframes as image references where possible."}),
                "seed_strategy": (["derived", "deterministic", "random"], {"default": "derived"}),
                "base_seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1}),
                "chunk_duration_seconds": ("FLOAT", {"default": 45.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "max_chunks": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "resume": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "final_concat": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("YVANN_EXECUTION_SETTINGS", "STRING")
    RETURN_NAMES = ("execution_config", "execution_summary")
    FUNCTION = "build"
    CATEGORY = "Yvann/Longform"

    def build(self, yvann_workflow_path, yvann_output_node_title, yvann_audio_analysis_mode, image_backend, continuity_mode, seed_strategy, base_seed, chunk_duration_seconds, max_chunks, resume, overwrite, final_concat):
        payload = {
            "workflow_template_path": str(_resolve_path(str(yvann_workflow_path))),
            "yvann_output_node_title": str(yvann_output_node_title),
            "yvann_audio_analysis_mode": str(yvann_audio_analysis_mode),
            "image_backend": str(image_backend),
            "continuity_mode": str(continuity_mode),
            "seed_strategy": str(seed_strategy),
            "base_seed": int(base_seed),
            "chunk_duration_seconds": float(chunk_duration_seconds),
            "max_chunks": int(max_chunks) if int(max_chunks) > 0 else None,
            "resume": bool(resume),
            "overwrite": bool(overwrite),
            "final_concat": bool(final_concat),
        }
        summary = f"{image_backend}, {continuity_mode}, {yvann_audio_analysis_mode}, chunks {chunk_duration_seconds}s, output '{yvann_output_node_title}'"
        return (json.dumps(payload, ensure_ascii=True), summary)


class LongformYvannAudioAnalysisPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cue_sheet_text": ("STRING", {"multiline": True, "default": ""}),
                "audio_source": ("YVANN_AUDIO_SOURCE",),
                "chunk_duration_seconds": ("FLOAT", {"default": 45.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "max_chunks": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("audio_waveform", "analysis_report")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Yvann/Longform"

    def preview(self, cue_sheet_text, audio_source, chunk_duration_seconds, max_chunks):
        audio_config = _decode_json_object(audio_source, "audio_source")
        audio_path = Path(str(audio_config.get("audio_path", "")))
        if not audio_path.exists():
            placeholder = torch.zeros((1, 240, 1280, 3), dtype=torch.float32)
            text = f"Audio file not found: {audio_path}"
            return {"ui": {"text": [text]}, "result": (placeholder, text)}
        duration = _audio_duration(audio_path)
        cues = _extract_cues(str(cue_sheet_text), duration)
        total_duration = duration or (float(cues[-1]["end"]) if cues else 0.0)
        chunks = _plan_chunks(total_duration, [float(cue["start"]) for cue in cues], float(chunk_duration_seconds), int(max_chunks)) if total_duration else []
        preview_path = _repo_root() / "output" / "yvann_audio_analysis" / f"waveform_{audio_path.stem}_{int(time.time())}.png"
        try:
            _write_audio_waveform_preview(audio_path, preview_path)
            tensor = _image_to_tensor(preview_path).unsqueeze(0)
            image_entry = _image_ui_entry(preview_path)
        except Exception as exc:
            tensor = torch.zeros((1, 240, 1280, 3), dtype=torch.float32)
            image_entry = None
            cues = cues or []
            total_duration = total_duration or 0.0
            extra_error = f"Waveform preview failed: {exc}"
        else:
            extra_error = ""
        lines = [
            f"Audio: {audio_path}",
            f"Duration: {_sec_to_hms(total_duration) if total_duration else 'unknown'}",
            f"Visual cue batches: {len(cues)}",
            f"Planned chunks: {len(chunks)}",
        ]
        if extra_error:
            lines.append(extra_error)
        for cue in cues[:24]:
            lines.append(f"{cue['id']} {_sec_to_hms(float(cue['start']))}-{_sec_to_hms(float(cue['end']))}: {str(cue['summary'])[:140]}")
        ui: dict[str, object] = {"text": lines}
        if image_entry:
            ui["images"] = [image_entry]
        return {"ui": ui, "result": (tensor, "\n".join(lines))}


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
                    {"multiline": False, "default": "custom_nodes/comfyui_yvann-nodes/example_workflows/AudioReactive_ImagesToVideo_Yvann.json"},
                ),
                "chunk_duration_seconds": ("FLOAT", {"default": 45.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "overlap_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3599.0, "step": 1.0}),
                "render_profile": (["draft", "balanced", "final", "custom"], {"default": "balanced"}),
                "image_interval_seconds": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 600.0, "step": 1.0}),
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
                "yvann_render_fps": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "yvann_min_frames": ("INT", {"default": 24, "min": 8, "max": 1024, "step": 1}),
                "yvann_max_frames": ("INT", {"default": 192, "min": 8, "max": 2048, "step": 1}),
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

    @classmethod
    def VALIDATE_INPUTS(cls, render_profile=None):
        _normalize_render_profile(render_profile)
        return True

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
            "render_profile": _normalize_render_profile(render_profile),
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
            "final_width": 1280,
            "final_height": 720,
            "final_fps": 24.0,
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
        process_record = {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "pid": proc.pid,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json(_process_record_path(config_path), process_record)

        JOB_REGISTRY[job_id] = {
            **process_record,
            "status": "running",
        }

        return (job_id, str(job_dir), str(config_path))


class LongformYvannCueSheetLauncher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "launch_now": ("BOOLEAN", {"default": True, "tooltip": "Start a backend longform job when this node is queued. Set false while editing."}),
                "cue_sheet_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "Paste the track list or timeline. Add visual scene markers in comments, for example: # A. 00:00:00 Rocket launch.",
                        "default": (
                            "00:00:00  1 Deep Hertz - Melting Sun  # A. 00:00:00 Rocket preparing for launch. "
                            "Close-ups of the rocket, smoke and ice falling.\n"
                            "00:04:39  2 Miguel Montero - Captain Hook  # B. 00:03:30 Rocket taking off, "
                            "climbing, stage separation."
                        ),
                    },
                ),
                "global_style_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "Style language applied to every generated keyframe to keep the longform output coherent.",
                        "default": "cinematic audio-reactive visuals, high detail, coherent motion, immersive lighting",
                    },
                ),
                "output_root": ("STRING", {"multiline": False, "default": "output/longform_yvann", "tooltip": "Backend job folders are written here, including job_state.json, images, videos, and final output."}),
                "audio_config": ("YVANN_AUDIO_SOURCE", {"tooltip": "Connect Yvann Longform Audio Source. This is where users upload/select audio."}),
                "profile_config": ("YVANN_RENDER_PROFILE", {"tooltip": "Connect Yvann Longform Render Profile."}),
                "execution_config": ("YVANN_EXECUTION_SETTINGS", {"tooltip": "Connect Yvann Longform Execution Settings."}),
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
        global_style_prompt,
        output_root,
        audio_config,
        profile_config,
        execution_config,
    ):
        if not bool(launch_now):
            return ("launch_disabled", "", "")

        if not str(cue_sheet_text).strip():
            raise ValueError("cue_sheet_text is empty")

        repo_root = _repo_root()
        resolved_output_root = _resolve_path(output_root)
        resolved_output_root.mkdir(parents=True, exist_ok=True)

        audio_settings = _json_config(audio_config)
        profile_settings = _json_config(profile_config)
        execution_settings = _json_config(execution_config)
        audio_path = str(audio_settings.get("audio_path") or "").strip()
        if not audio_path:
            raise ValueError("Connect a Yvann Longform Audio Source node before launching.")

        job_id = _job_id()
        job_dir = resolved_output_root / job_id
        config_path = _config_path(resolved_output_root, job_id)
        cue_sheet_path = config_path.with_suffix(".cuesheet.txt")
        _write_text(cue_sheet_path, str(cue_sheet_text).strip() + "\n")
        audio_source_path = _resolve_path(audio_path)

        render_profile = profile_settings.get("render_profile", "balanced")

        config = {
            "job_id": job_id,
            "script_path": str(cue_sheet_path),
            "audio_path": str(audio_source_path),
            "global_style_prompt": global_style_prompt,
            "output_root": str(resolved_output_root),
            "comfy_api_url": "http://127.0.0.1:18188",
            "comfy_api_verify_tls": False,
            "workflow_template_path": str(_resolve_path(str(execution_settings.get("workflow_template_path", "custom_nodes/comfyui_yvann-nodes/example_workflows/AudioReactive_ImagesToVideo_Yvann.json")))),
            "comfy_root": str(repo_root),
            "chunk_duration_seconds": float(execution_settings.get("chunk_duration_seconds", 45.0)),
            "overlap_seconds": 0.0,
            "segmentation_mode": "auto",
            "motifs": [],
            "negative_prompt": "low quality, blurry, watermark, text artifacts",
            "continuity_mode": str(execution_settings.get("continuity_mode", "style")),
            "render_profile": _normalize_render_profile(render_profile),
            "image_backend": str(execution_settings.get("image_backend", "comfy_api")),
            "profile_behavior": str(profile_settings.get("profile_behavior") or _profile_override_note(render_profile)),
            "image_interval_seconds": float(profile_settings.get("image_interval_seconds", 6.0)),
            "image_width": int(profile_settings.get("image_width", 1280)),
            "image_height": int(profile_settings.get("image_height", 720)),
            "comfy_t2i_checkpoint": "DreamShaper_8_pruned.safetensors",
            "comfy_t2i_steps": int(profile_settings.get("comfy_t2i_steps", 10)),
            "comfy_t2i_cfg": float(profile_settings.get("comfy_t2i_cfg", 4.5)),
            "comfy_t2i_sampler": "euler",
            "comfy_t2i_scheduler": "normal",
            "seed_strategy": str(execution_settings.get("seed_strategy", "derived")),
            "base_seed": int(execution_settings.get("base_seed", 42)),
            "seed_offset": 1009,
            "resume": bool(execution_settings.get("resume", True)),
            "overwrite": bool(execution_settings.get("overwrite", False)),
            "stop_on_failure": False,
            "final_concat": bool(execution_settings.get("final_concat", True)),
            "ffmpeg_video_codec": "libx264",
            "ffmpeg_crf": int(profile_settings.get("ffmpeg_crf", 22)),
            "final_width": 1280,
            "final_height": 720,
            "final_fps": 24.0,
            "yvann_output_node_title": str(execution_settings.get("yvann_output_node_title", "First Pass | Low Res")),
            "yvann_audio_analysis_mode": str(execution_settings.get("yvann_audio_analysis_mode", "Full Audio")),
            "yvann_render_fps": float(profile_settings.get("yvann_render_fps", 6.0)),
            "yvann_min_frames": int(profile_settings.get("yvann_min_frames", 24)),
            "yvann_max_frames": int(profile_settings.get("yvann_max_frames", 192)),
            "max_chunks": execution_settings.get("max_chunks"),
        }

        log_path = job_dir / "launcher.log"
        _write_config(config_path, config)
        proc = _launch_process(config_path, log_path)
        process_record = {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "config_path": str(config_path),
            "cue_sheet_path": str(cue_sheet_path),
            "log_path": str(log_path),
            "pid": proc.pid,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json(_process_record_path(config_path), process_record)

        JOB_REGISTRY[job_id] = {
            **process_record,
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
                "job_id": ("STRING", {"multiline": False, "default": "", "tooltip": "Job id returned by the launcher. Leave empty to show the latest backend job."}),
                "job_dir": ("STRING", {"multiline": False, "default": "", "tooltip": "Backend job directory returned by the launcher. Leave empty to show the latest backend job."}),
                "config_path": ("STRING", {"multiline": False, "default": "", "tooltip": "Launcher config path returned by the launcher; used for display only."}),
            },
            "optional": {
                "output_root": ("STRING", {"multiline": False, "default": "output/longform_yvann", "tooltip": "Folder to scan for persisted backend job_state.json files."}),
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
                    f"backend status: {state.get('status', 'running')}; "
                    f"progress: {len(state.get('completed_chunks', []))}/{state.get('number_of_chunks', '?')} chunks completed; "
                    f"failed: {len(state.get('failed_chunks', []))}; concat: {state.get('final_concat_status')}; "
                    f"cancel requested: {bool(state.get('cancel_requested')) or (resolved_job_dir / 'cancel.requested').exists()}"
                )
                if state.get("render_profile"):
                    lines.append(f"render profile: {state.get('render_profile')} - {state.get('render_profile_behavior', 'profile preset applied')}")
                if state.get("yvann_audio_analysis_mode") or state.get("yvann_output_node_title"):
                    lines.append(
                        f"Yvann execution: output '{state.get('yvann_output_node_title', 'First Pass | Low Res')}', "
                        f"audio analysis {state.get('yvann_audio_analysis_mode', 'Full Audio')}"
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


class LongformYvannCancelJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cancel_now": ("BOOLEAN", {"default": False, "tooltip": "Set true and queue this node to request backend cancellation."}),
                "job_id": ("STRING", {"multiline": False, "default": "", "tooltip": "Job id to cancel. Leave empty to cancel the latest discovered backend job."}),
                "job_dir": ("STRING", {"multiline": False, "default": "", "tooltip": "Job directory to cancel. If blank, the node finds the job by job_id or latest output."}),
            },
            "optional": {
                "output_root": ("STRING", {"multiline": False, "default": "output/longform_yvann", "tooltip": "Folder to scan when job_id/job_dir are empty."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cancel_status",)
    FUNCTION = "cancel"
    OUTPUT_NODE = True
    CATEGORY = "Yvann/Longform"

    def cancel(self, cancel_now, job_id, job_dir, output_root="output/longform_yvann"):
        if not bool(cancel_now):
            text = "cancel_now is false; no backend job was changed."
            return {"ui": {"text": [text]}, "result": (text,)}

        resolved_job_dir = Path(str(job_dir).strip()) if str(job_dir).strip() else None
        if resolved_job_dir and not resolved_job_dir.is_absolute():
            resolved_job_dir = (_repo_root() / resolved_job_dir).resolve()

        resolved_job_id = str(job_id).strip()
        if not resolved_job_dir or not resolved_job_id:
            job = _find_job(resolved_job_id, _resolve_path(str(output_root)) if str(output_root).strip() else None)
            if job:
                resolved_job_id = str(job.get("job_id"))
                resolved_job_dir = Path(str(job.get("job_dir")))

        if not resolved_job_dir or not resolved_job_id:
            text = "No backend job found to cancel. Provide job_id/job_dir from the launcher or status dashboard."
            return {"ui": {"text": [text]}, "result": (text,)}

        result = _request_cancel(resolved_job_id, resolved_job_dir)
        text = f"Cancellation requested for backend job {result['job_id']}. The runner will stop at the current safe checkpoint."
        return {"ui": {"text": [text, f"cancel_path: {result['cancel_path']}"]}, "result": (text,)}


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
    jobs = _discover_jobs()
    return web.json_response({"jobs": jobs})


@PromptServer.instance.routes.post("/yvann_longform/jobs/{job_id}/cancel")
async def cancel_job(request):
    job_id = str(request.match_info.get("job_id", "")).strip()
    job = _find_job(job_id)
    if not job:
        return web.json_response({"error": f"Job not found: {job_id}"}, status=404)
    result = _request_cancel(job_id, Path(str(job["job_dir"])))
    return web.json_response(result)


NODE_CLASS_MAPPINGS = {
    "LongformYvannLauncher": LongformYvannLauncher,
    "LongformYvannCueSheetSource": LongformYvannCueSheetSource,
    "LongformYvannRenderProfile": LongformYvannRenderProfile,
    "LongformYvannAudioSource": LongformYvannAudioSource,
    "LongformYvannExecutionSettings": LongformYvannExecutionSettings,
    "LongformYvannAudioAnalysisPreview": LongformYvannAudioAnalysisPreview,
    "LongformYvannCueSheetLauncher": LongformYvannCueSheetLauncher,
    "LongformYvannCueSheetParser": LongformYvannCueSheetParser,
    "LongformYvannCueSheetBatchPlan": LongformYvannCueSheetBatchPlan,
    "LongformYvannJobStatus": LongformYvannJobStatus,
    "LongformYvannCancelJob": LongformYvannCancelJob,
    "LongformYvannGeneratedImagesOutput": LongformYvannGeneratedImagesOutput,
    "LongformYvannFourImagesOutput": LongformYvannFourImagesOutput,
    "LongformYvannWorkflowInspector": LongformYvannWorkflowInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongformYvannLauncher": "Yvann Longform Launcher",
    "LongformYvannCueSheetSource": "Yvann Longform Cue Sheet",
    "LongformYvannRenderProfile": "Yvann Longform Render Profile",
    "LongformYvannAudioSource": "Yvann Longform Audio Source",
    "LongformYvannExecutionSettings": "Yvann Longform Execution Settings",
    "LongformYvannAudioAnalysisPreview": "Yvann Longform Audio Analysis Preview",
    "LongformYvannCueSheetLauncher": "Yvann Longform Image-to-Video",
    "LongformYvannCueSheetParser": "Yvann Longform Cue Sheet Parser",
    "LongformYvannCueSheetBatchPlan": "Yvann Longform Batch Plan",
    "LongformYvannJobStatus": "Yvann Longform Job Status",
    "LongformYvannCancelJob": "Yvann Longform Cancel Job",
    "LongformYvannGeneratedImagesOutput": "Yvann Longform Generated Images",
    "LongformYvannFourImagesOutput": "Yvann Longform Scene Batch",
    "LongformYvannWorkflowInspector": "Yvann Workflow Inspector",
}
