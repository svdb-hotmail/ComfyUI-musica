#!/usr/bin/env python3
"""
Long-form scripted image-to-video orchestration for Yvann workflows.

This runner is additive and does not modify existing workflows. It builds a
chunk manifest from script + audio, generates per-chunk images, renders each
chunk with a converted Yvann workflow over ComfyUI API, tracks persistent job
state, supports resume, and optionally concatenates chunk outputs.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: requests. Install with `pip install requests`.") from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pillow. Install with `pip install pillow`.") from exc


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sec_to_hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    ms = int(round((seconds - whole) * 1000.0))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def hms_to_sec(value: str) -> float:
    value = value.strip()
    parts = value.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError(f"Unsupported timestamp format: {value}")


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


@dataclasses.dataclass
class JobConfig:
    script_path: str
    audio_path: str
    global_style_prompt: str
    output_root: str

    comfy_api_url: str = "http://127.0.0.1:18188"
    comfy_api_verify_tls: bool = False
    workflow_template_path: str = "custom_nodes/comfyui_yvann-nodes/example_workflows/AudioReactive_ImagesToVideo_Yvann.json"
    comfy_root: str = "."

    chunk_duration_seconds: float = 30.0
    overlap_seconds: float = 0.0
    segmentation_mode: str = "auto"  # auto | fixed | timestamped | scene

    motifs: list[str] | None = None
    negative_prompt: str = "low quality, blurry, watermark, text artifacts"
    continuity_mode: str = "style"  # independent | style | carry

    image_backend: str = "procedural"  # procedural | comfy_api
    images_per_chunk: int = 1
    image_interval_seconds: float = 5.0
    image_width: int = 1024
    image_height: int = 576

    comfy_t2i_checkpoint: str = "DreamShaper_8_pruned.safetensors"
    comfy_t2i_steps: int = 12
    comfy_t2i_cfg: float = 5.0
    comfy_t2i_sampler: str = "euler"
    comfy_t2i_scheduler: str = "normal"

    seed_strategy: str = "derived"  # deterministic | derived | random
    base_seed: int = 42
    seed_offset: int = 1009

    resume: bool = True
    overwrite: bool = False
    stop_on_failure: bool = False
    final_concat: bool = True
    ffmpeg_video_codec: str = "libx264"
    ffmpeg_crf: int = 18
    resume_job_dir: str | None = None

    yvann_output_node_title: str = "First Pass | Low Res"
    yvann_render_fps: float = 8.0
    yvann_min_frames: int = 24
    yvann_max_frames: int = 192
    job_id: str | None = None

    # Optional cap for tests/sampling.
    max_chunks: int | None = None


@dataclasses.dataclass
class Chunk:
    chunk_id: str
    index: int
    start_time: float
    end_time: float
    chunk_duration: float
    scene_summary: str
    scene_prompt: str
    negative_prompt: str
    visual_theme_tags: list[str]
    image_generation_prompts: list[str]
    assigned_image_paths: list[str]
    audio_chunk_path: str
    video_chunk_path: str
    status: str = "pending"
    error: str | None = None
    visual_batch_id: str | None = None
    visual_batch_start: float | None = None
    visual_batch_end: float | None = None
    visual_batch_image_paths: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class VisualCue:
    cue_id: str
    start_time: float
    end_time: float
    summary: str

    def overlaps(self, start: float, end: float) -> bool:
        return not (self.end_time <= start or self.start_time >= end)

    def contains_midpoint(self, start: float, end: float) -> bool:
        midpoint = (start + end) * 0.5
        return self.start_time <= midpoint < self.end_time


class ComfyClient:
    def __init__(self, base_url: str, verify_tls: bool = False):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.verify_tls = verify_tls

    def get_json(self, path: str) -> dict[str, Any]:
        r = self.session.get(f"{self.base_url}{path}", timeout=30, verify=self.verify_tls)
        r.raise_for_status()
        return r.json()

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=120,
            verify=self.verify_tls,
        )
        if r.status_code >= 400:
            raise RuntimeError(
                f"HTTP {r.status_code} on {path}: {r.text[:4000]}"
            )
        return r.json()

    def queue_prompt(self, prompt: dict[str, Any], partial_targets: list[str] | None = None) -> str:
        payload: dict[str, Any] = {"prompt": prompt}
        if partial_targets:
            payload["partial_execution_targets"] = partial_targets
        resp = self.post_json("/prompt", payload)
        return str(resp["prompt_id"])

    def wait_for_completion(self, prompt_id: str, poll_seconds: float = 2.0) -> dict[str, Any]:
        while True:
            history = self.get_json(f"/history/{prompt_id}")
            if history and prompt_id in history:
                item = history[prompt_id]
                status = item.get("status", {})
                status_str = status.get("status_str")
                if status_str == "error":
                    errors = item.get("node_errors", {})
                    raise RuntimeError(f"Prompt failed: {errors}")
                return item
            time.sleep(poll_seconds)

    def convert_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        return self.post_json("/workflow/convert", workflow)


class LongformYvannRunner:
    def __init__(self, config: JobConfig):
        self.config = config
        self.comfy_root = Path(config.comfy_root).resolve()
        self.script_path = Path(config.script_path).resolve()
        self.audio_path = Path(config.audio_path).resolve()
        self.workflow_template_path = Path(config.workflow_template_path)
        if not self.workflow_template_path.is_absolute():
            self.workflow_template_path = (self.comfy_root / self.workflow_template_path).resolve()

        self.job_id = config.job_id or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.job_dir = Path(config.output_root).resolve() / self.job_id
        self.manifest_dir = self.job_dir / "manifest"
        self.audio_chunks_dir = self.job_dir / "audio_chunks"
        self.images_dir = self.job_dir / "images"
        self.videos_dir = self.job_dir / "videos"
        self.previews_dir = self.job_dir / "previews"
        self.final_dir = self.job_dir / "final"

        self.job_config_path = self.job_dir / "job_config.json"
        self.job_state_path = self.job_dir / "job_state.json"
        self.chunk_manifest_path = self.manifest_dir / "chunk_manifest.json"

        self._audio_duration: float | None = None
        self.client = ComfyClient(config.comfy_api_url, verify_tls=config.comfy_api_verify_tls)

    def _set_job_paths_from_dir(self, job_dir: Path) -> None:
        self.job_dir = job_dir.resolve()
        self.job_id = self.job_dir.name
        self.manifest_dir = self.job_dir / "manifest"
        self.audio_chunks_dir = self.job_dir / "audio_chunks"
        self.images_dir = self.job_dir / "images"
        self.videos_dir = self.job_dir / "videos"
        self.previews_dir = self.job_dir / "previews"
        self.final_dir = self.job_dir / "final"
        self.job_config_path = self.job_dir / "job_config.json"
        self.job_state_path = self.job_dir / "job_state.json"
        self.chunk_manifest_path = self.manifest_dir / "chunk_manifest.json"

    def _find_resumable_job(self) -> Path | None:
        if self.config.resume_job_dir:
            p = Path(self.config.resume_job_dir).resolve()
            return p if p.exists() else None

        root = Path(self.config.output_root).resolve()
        if not root.exists():
            return None

        candidates = sorted(root.glob("job_*"), reverse=True)
        for c in candidates:
            state_path = c / "job_state.json"
            manifest_path = c / "manifest" / "chunk_manifest.json"
            if not state_path.exists() or not manifest_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if Path(state.get("source_script_path", "")).resolve() != self.script_path:
                    continue
                if Path(state.get("source_audio_path", "")).resolve() != self.audio_path:
                    continue
                total = int(state.get("number_of_chunks") or 0)
                completed = len(state.get("completed_chunks") or [])
                if completed < total:
                    return c
            except Exception:
                continue
        return None

    @staticmethod
    def _chunks_from_manifest(manifest_payload: dict[str, Any]) -> list[Chunk]:
        chunks = []
        for entry in manifest_payload.get("chunks", []):
            chunks.append(Chunk(**entry))
        return chunks

    def validate(self) -> None:
        if not self.script_path.exists():
            raise FileNotFoundError(f"script_path does not exist: {self.script_path}")
        if not self.audio_path.exists():
            raise FileNotFoundError(f"audio_path does not exist: {self.audio_path}")
        if not self.workflow_template_path.exists():
            raise FileNotFoundError(f"workflow_template_path does not exist: {self.workflow_template_path}")
        if self.config.chunk_duration_seconds <= 0:
            raise ValueError("chunk_duration_seconds must be > 0")
        if self.config.overlap_seconds < 0:
            raise ValueError("overlap_seconds must be >= 0")
        if self.config.overlap_seconds >= self.config.chunk_duration_seconds:
            raise ValueError("overlap_seconds must be less than chunk_duration_seconds")
        if self.config.image_interval_seconds <= 0:
            raise ValueError("image_interval_seconds must be > 0")

        run_cmd(["ffmpeg", "-version"])  # validates ffmpeg availability
        run_cmd(["ffprobe", "-version"])  # validates ffprobe availability

        self._audio_duration = self.get_audio_duration(self.audio_path)
        if self._audio_duration <= 1.0:
            raise ValueError("Audio duration looks invalid (<= 1 second)")

        est_step = max(1.0, self.config.chunk_duration_seconds - self.config.overlap_seconds)
        est_chunks = int(math.ceil(self._audio_duration / est_step))
        if est_chunks > 20000:
            raise ValueError(f"Chunk count too high ({est_chunks}); adjust duration/overlap")

        self.job_dir.mkdir(parents=True, exist_ok=True)
        test_file = self.job_dir / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()

        # Validate required nodes for this runner mode.
        expected = [
            "comfyui_yvann-nodes",
            "comfyui-videohelpersuite",
            "comfyui-animatediff-evolved",
            "comfyui_ipadapter_plus",
            "comfyui-advanced-controlnet",
            "comfyui_controlnet_aux",
            "comfyui-kjnodes",
            "comfyui-workflow-to-api-converter-endpoint",
        ]
        for name in expected:
            p = self.comfy_root / "custom_nodes" / name
            if not p.exists():
                raise FileNotFoundError(f"Required custom node missing: {p}")

        # Validate API endpoints.
        self.client.get_json("/system_stats")
        self.client.get_json("/workflow_templates")

    @staticmethod
    def get_audio_duration(audio_path: Path) -> float:
        proc = run_cmd(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ]
        )
        return float(proc.stdout.strip())

    def load_script_text(self) -> str:
        return self.script_path.read_text(encoding="utf-8", errors="replace")

    def _extract_timed_sections(self, script_text: str) -> list[tuple[float, float, str]]:
        lines = [ln.rstrip() for ln in script_text.splitlines()]
        out: list[tuple[float, float, str]] = []
        current: list[str] = []
        current_start: float | None = None
        current_end: float | None = None

        pattern = re.compile(
            r"^\s*(?P<start>\d{1,2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?)\s*(?:-|to|->|–)\s*(?P<end>\d{1,2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?)\s*(?::\s*(?P<rest>.*))?$",
            re.IGNORECASE,
        )

        for line in lines:
            m = pattern.match(line)
            if m:
                if current_start is not None:
                    out.append((current_start, current_end or current_start, "\n".join(current).strip()))
                current = []
                current_start = hms_to_sec(m.group("start"))
                current_end = hms_to_sec(m.group("end"))
                rest = (m.group("rest") or "").strip()
                if rest:
                    current.append(rest)
            else:
                if current_start is not None:
                    current.append(line)

        if current_start is not None:
            out.append((current_start, current_end or current_start, "\n".join(current).strip()))

        return [x for x in out if x[1] > x[0]]

    def _extract_visual_cues(self, script_text: str, total_duration: float) -> list[VisualCue]:
        lines = [ln.rstrip() for ln in script_text.splitlines()]
        marker_pattern = re.compile(
            r"#\s*(?:(?P<label>[A-Z])\.\s*)?(?P<time>\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s*(?P<text>.*)$",
            re.IGNORECASE,
        )
        continuation_pattern = re.compile(r"^\s*#\s*(?P<text>.+?)\s*$")

        markers: list[tuple[str, float, list[str]]] = []
        for line in lines:
            marker = marker_pattern.search(line)
            if marker:
                label = (marker.group("label") or f"cue_{len(markers) + 1:02d}").upper()
                start_time = hms_to_sec(marker.group("time"))
                text = marker.group("text").strip()
                markers.append((label, start_time, [text] if text else []))
                continue

            continuation = continuation_pattern.match(line)
            if continuation and markers:
                text = continuation.group("text").strip()
                if text:
                    markers[-1][2].append(text)

        cues: list[VisualCue] = []
        for idx, (label, start_time, parts) in enumerate(markers):
            if start_time >= total_duration:
                continue
            next_start = total_duration
            for _next_label, candidate_start, _next_parts in markers[idx + 1:]:
                if candidate_start > start_time:
                    next_start = min(candidate_start, total_duration)
                    break
            summary = " ".join(" ".join(parts).split())
            if not summary:
                continue
            cues.append(
                VisualCue(
                    cue_id=label,
                    start_time=max(0.0, start_time),
                    end_time=max(start_time, next_start),
                    summary=summary,
                )
            )

        return [cue for cue in cues if cue.end_time > cue.start_time]

    def _split_sections(self, script_text: str) -> list[str]:
        blocks = [blk.strip() for blk in re.split(r"\n\s*\n", script_text) if blk.strip()]
        if not blocks:
            return ["Abstract visual storytelling progression."]
        return blocks

    def _plan_chunk_boundaries(self, total_duration: float, split_points: list[float] | None = None) -> list[tuple[float, float]]:
        step = self.config.chunk_duration_seconds - self.config.overlap_seconds
        cue_splits = sorted({round(p, 3) for p in (split_points or []) if 0.0 < p < total_duration})
        boundaries: list[tuple[float, float]] = []
        t = 0.0
        idx = 0
        while t < total_duration:
            end = min(total_duration, t + self.config.chunk_duration_seconds)
            for split in cue_splits:
                if t + 0.001 < split < end - 0.001:
                    end = split
                    break
            boundaries.append((t, end))
            idx += 1
            if self.config.max_chunks is not None and idx >= self.config.max_chunks:
                break
            if end >= total_duration:
                break
            if any(abs(end - split) <= 0.001 for split in cue_splits):
                t = end
            else:
                t += step
        return boundaries

    @staticmethod
    def _find_visual_cue(cues: list[VisualCue], start: float, end: float) -> VisualCue | None:
        for cue in cues:
            if cue.contains_midpoint(start, end):
                return cue
        for cue in cues:
            if cue.overlaps(start, end):
                return cue
        return None

    def _energy_tag(self, frac: float) -> str:
        if frac < 0.2:
            return "slow_buildup"
        if frac < 0.45:
            return "rising_intensity"
        if frac < 0.7:
            return "high_energy"
        if frac < 0.9:
            return "cinematic_climax"
        return "cooldown_afterglow"

    def _build_prompt(self, summary: str, frac: float, previous_summary: str | None) -> str:
        motifs = ", ".join(self.config.motifs or [])
        continuity = ""
        if self.config.continuity_mode == "carry" and previous_summary:
            continuity = f" continuity with prior scene: {previous_summary[:180]}"
        elif self.config.continuity_mode == "style":
            continuity = " preserve consistent style language with previous chunk"

        energy = self._energy_tag(frac)
        parts = [
            self.config.global_style_prompt.strip(),
            f"scene: {summary[:500]}",
            f"energy: {energy}",
        ]
        if motifs:
            parts.append(f"motifs: {motifs}")
        if continuity:
            parts.append(continuity.strip())
        return ", ".join([p for p in parts if p])

    def _batch_dir_for(self, batch_id: str | None) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", batch_id or "unbatched").strip("_") or "unbatched"
        return self.images_dir / safe

    def _image_count_for_duration(self, duration: float) -> int:
        return max(1, int(math.ceil(duration / max(self.config.image_interval_seconds, 1e-6))))

    @staticmethod
    def _vae_safe_dimension(value: int) -> int:
        return max(64, int(math.ceil(max(1, value) / 8) * 8))

    def _variation_prompt(self, base_prompt: str, batch_id: str | None, image_index: int, count: int) -> str:
        phase = image_index / max(count - 1, 1)
        camera = [
            "wide establishing view",
            "medium cinematic composition",
            "close-up detail shot",
            "low-angle dramatic view",
            "aerial tracking view",
            "macro texture detail",
        ][(image_index - 1) % 6]
        lighting = [
            "soft atmospheric light",
            "high contrast rim light",
            "volumetric beams",
            "reflected colored light",
            "deep shadow gradients",
            "glowing practical lights",
        ][(image_index - 1) % 6]
        motion = [
            "designed as a video keyframe with clear directional motion",
            "progressive transformation from the previous image",
            "environment evolving with new foreground elements",
            "background reveals more depth and scale",
            "subject shifts position while preserving scene continuity",
        ][(image_index - 1) % 5]
        return (
            f"{base_prompt}, visual batch {batch_id or 'unbatched'}, image {image_index} of {count}, "
            f"timeline phase {phase:.2f}, {camera}, {lighting}, {motion}, no text, no watermark"
        )

    def build_manifest(self) -> list[Chunk]:
        script_text = self.load_script_text()
        audio_duration = self._audio_duration or self.get_audio_duration(self.audio_path)

        visual_cues = self._extract_visual_cues(script_text, audio_duration)
        timed_sections = self._extract_timed_sections(script_text)
        split_points = [cue.start_time for cue in visual_cues]
        boundaries = self._plan_chunk_boundaries(audio_duration, split_points=split_points)
        sections = self._split_sections(script_text)

        chunks: list[Chunk] = []
        prev_summary: str | None = None
        for i, (start, end) in enumerate(boundaries, start=1):
            frac = ((start + end) * 0.5) / max(audio_duration, 1e-6)
            visual_cue = self._find_visual_cue(visual_cues, start, end)

            if visual_cue and self.config.segmentation_mode in {"auto", "timestamped", "cue_sheet"}:
                summary = visual_cue.summary
            elif timed_sections and self.config.segmentation_mode in {"auto", "timestamped"}:
                matching = [x[2] for x in timed_sections if not (x[1] <= start or x[0] >= end)]
                summary = " ".join(matching).strip() or sections[min(i - 1, len(sections) - 1)]
            else:
                section_idx = min(int(frac * len(sections)), len(sections) - 1)
                summary = sections[section_idx]

            summary = " ".join(summary.split())
            if visual_cue:
                summary = f"Visual batch {visual_cue.cue_id}: {summary}"
            scene_prompt = self._build_prompt(summary, frac, prev_summary)
            prev_summary = summary

            chunk_id = f"chunk_{i:04d}"
            chunk = Chunk(
                chunk_id=chunk_id,
                index=i,
                start_time=start,
                end_time=end,
                chunk_duration=end - start,
                scene_summary=summary,
                scene_prompt=scene_prompt,
                negative_prompt=self.config.negative_prompt,
                visual_theme_tags=[self._energy_tag(frac)] + (self.config.motifs or []),
                image_generation_prompts=[scene_prompt],
                assigned_image_paths=[],
                audio_chunk_path=str(self.audio_chunks_dir / f"{chunk_id}.wav"),
                video_chunk_path=str(self.videos_dir / f"{chunk_id}.mp4"),
                status="planned",
                visual_batch_id=visual_cue.cue_id if visual_cue else None,
                visual_batch_start=visual_cue.start_time if visual_cue else None,
                visual_batch_end=visual_cue.end_time if visual_cue else None,
                visual_batch_image_paths=[],
            )
            chunks.append(chunk)

        return chunks

    def _seed_for(self, chunk_index: int, variant_index: int, prompt: str) -> int:
        if self.config.seed_strategy == "random":
            return random.randint(0, 2_147_483_647)
        if self.config.seed_strategy == "deterministic":
            return self.config.base_seed + chunk_index * self.config.seed_offset + variant_index
        # derived
        digest = hashlib.sha256(f"{self.config.base_seed}|{chunk_index}|{variant_index}|{prompt}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _write_job_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.job_state_path, state)

    def _write_manifest(self, chunks: list[Chunk]) -> None:
        payload = {
            "job_id": self.job_id,
            "created_at": now_utc(),
            "source_script_path": str(self.script_path),
            "source_audio_path": str(self.audio_path),
            "workflow_template_path": str(self.workflow_template_path),
            "chunks": [c.as_dict() for c in chunks],
        }
        atomic_write_json(self.chunk_manifest_path, payload)

    def _copy_job_sources(self) -> None:
        (self.job_dir / "script_source.txt").write_text(self.load_script_text(), encoding="utf-8")
        shutil.copy2(self.audio_path, self.job_dir / self.audio_path.name)

    def _initial_state(self, chunks: list[Chunk]) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "created_at": now_utc(),
            "updated_at": now_utc(),
            "source_script_path": str(self.script_path),
            "source_audio_path": str(self.audio_path),
            "total_audio_duration": self._audio_duration,
            "chunk_duration": self.config.chunk_duration_seconds,
            "overlap": self.config.overlap_seconds,
            "image_interval_seconds": self.config.image_interval_seconds,
            "number_of_chunks": len(chunks),
            "current_chunk_index": 0,
            "completed_chunks": [],
            "failed_chunks": [],
            "image_generation_status": {},
            "video_generation_status": {},
            "final_concat_status": "pending",
            "resume": self.config.resume,
            "timestamps": {"created_at": now_utc(), "last_update": now_utc()},
            "discovered_paths": {
                "comfy_root": str(self.comfy_root),
                "workflow_template": str(self.workflow_template_path),
                "script": str(self.script_path),
                "audio": str(self.audio_path),
            },
        }

    def prepare_job(self) -> tuple[list[Chunk], dict[str, Any]]:
        if self.config.resume:
            resumable = self._find_resumable_job()
            if resumable is not None:
                self._set_job_paths_from_dir(resumable)
                manifest_payload = json.loads(self.chunk_manifest_path.read_text(encoding="utf-8"))
                state = json.loads(self.job_state_path.read_text(encoding="utf-8"))
                chunks = self._chunks_from_manifest(manifest_payload)
                return chunks, state

        for d in [
            self.job_dir,
            self.manifest_dir,
            self.audio_chunks_dir,
            self.images_dir,
            self.videos_dir,
            self.previews_dir,
            self.final_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        chunks = self.build_manifest()
        state = self._initial_state(chunks)

        self._copy_job_sources()
        atomic_write_json(self.job_config_path, dataclasses.asdict(self.config))
        self._write_manifest(chunks)
        self._write_job_state(state)
        return chunks, state

    def split_audio_for_chunk(self, chunk: Chunk) -> None:
        out_path = Path(chunk.audio_chunk_path)
        if out_path.exists() and not self.config.overwrite:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                sec_to_hms(chunk.start_time),
                "-t",
                sec_to_hms(chunk.chunk_duration),
                "-i",
                str(self.audio_path),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(out_path),
            ]
        )

    def _generate_procedural_image(self, path: Path, prompt: str, seed: int) -> None:
        rng = random.Random(seed)
        w = self.config.image_width
        h = self.config.image_height
        img = Image.new("RGB", (w, h), color=(8, 10, 18))
        dr = ImageDraw.Draw(img)

        # Layered gradients and ribbons for stylized abstract animation keyframes.
        for _ in range(120):
            x1 = rng.randint(0, w)
            y1 = rng.randint(0, h)
            x2 = x1 + rng.randint(-300, 300)
            y2 = y1 + rng.randint(-220, 220)
            color = (
                rng.randint(20, 240),
                rng.randint(20, 240),
                rng.randint(20, 240),
            )
            width = rng.randint(1, 6)
            dr.line((x1, y1, x2, y2), fill=color, width=width)

        for _ in range(30):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            r = rng.randint(20, 180)
            color = (
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(40, 120),
            )
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.ellipse((x - r, y - r, x + r, y + r), fill=color)
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        dr = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        prompt_short = textwrap.shorten(prompt, width=120, placeholder="...")
        dr.rectangle((8, h - 60, w - 8, h - 8), fill=(0, 0, 0))
        dr.text((16, h - 48), prompt_short, fill=(245, 245, 245), font=font)
        img.save(path)

    def _generate_comfy_t2i_image(self, prompt: str, negative_prompt: str, seed: int) -> Path:
        # Minimal API prompt image generation workflow.
        out_prefix = f"longform_yvann/t2i/{self.job_id}/seed_{seed}"
        latent_width = self._vae_safe_dimension(self.config.image_width)
        latent_height = self._vae_safe_dimension(self.config.image_height)
        api_prompt = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": int(self.config.comfy_t2i_steps),
                    "cfg": float(self.config.comfy_t2i_cfg),
                    "sampler_name": self.config.comfy_t2i_sampler,
                    "scheduler": self.config.comfy_t2i_scheduler,
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": self.config.comfy_t2i_checkpoint},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": latent_width,
                    "height": latent_height,
                    "batch_size": 1,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["4", 1]},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["4", 1]},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": out_prefix, "images": ["8", 0]},
            },
        }

        prompt_id = self.client.queue_prompt(api_prompt, partial_targets=["9"])
        hist = self.client.wait_for_completion(prompt_id)
        outputs = hist.get("outputs", {}).get("9", {}).get("images", [])
        if not outputs:
            raise RuntimeError("No image output returned from comfy_t2i backend")

        first = outputs[0]
        # Comfy returns subfolder + filename relative to output root.
        output_root = self.comfy_root / "output"
        p = output_root / first.get("subfolder", "") / first["filename"]
        if not p.exists():
            raise FileNotFoundError(f"Generated image was not found: {p}")
        return p

    def generate_images_for_chunk(self, chunk: Chunk) -> None:
        if chunk.assigned_image_paths and chunk.visual_batch_image_paths and not self.config.overwrite:
            return

        batch_id = chunk.visual_batch_id or chunk.chunk_id
        batch_start = chunk.visual_batch_start if chunk.visual_batch_start is not None else chunk.start_time
        batch_end = chunk.visual_batch_end if chunk.visual_batch_end is not None else chunk.end_time
        batch_duration = max(batch_end - batch_start, chunk.chunk_duration)
        batch_count = self._image_count_for_duration(batch_duration)
        batch_dir = self._batch_dir_for(batch_id)
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_images: list[str] = []
        for image_number in range(1, batch_count + 1):
            prompt = self._variation_prompt(chunk.scene_prompt, batch_id, image_number, batch_count)
            seed = self._seed_for(chunk.index, image_number, prompt)
            dst = batch_dir / f"{batch_id}_{image_number:04d}.png"

            if dst.exists() and not self.config.overwrite:
                batch_images.append(str(dst))
                continue

            if self.config.image_backend == "comfy_api":
                try:
                    generated = self._generate_comfy_t2i_image(
                        prompt,
                        chunk.negative_prompt,
                        seed,
                    )
                    shutil.copy2(generated, dst)
                except Exception:
                    # Fallback to procedural to keep long jobs progressing.
                    self._generate_procedural_image(dst, prompt, seed)
            else:
                self._generate_procedural_image(dst, prompt, seed)

            batch_images.append(str(dst))

        chunk.visual_batch_image_paths = batch_images
        chunk.assigned_image_paths = batch_images

    def _load_and_convert_yvann_template(self) -> dict[str, Any]:
        workflow = json.loads(self.workflow_template_path.read_text(encoding="utf-8"))
        api_prompt = self.client.convert_workflow(workflow)
        self._normalize_prompt_values(api_prompt)
        self._normalize_available_combo_values(api_prompt)
        return api_prompt

    def _normalize_available_combo_values(self, prompt: dict[str, Any]) -> None:
        object_info = self.client.get_json("/object_info")
        preferred_values = {"ckpt_name": self.config.comfy_t2i_checkpoint}
        for node in prompt.values():
            class_type = node.get("class_type")
            class_info = object_info.get(class_type)
            if not class_info:
                continue
            input_specs = {}
            for section in ("required", "optional"):
                input_specs.update(class_info.get("input", {}).get(section, {}))
            inputs = node.setdefault("inputs", {})
            for input_name, value in list(inputs.items()):
                if isinstance(value, list) or input_name not in input_specs:
                    continue
                spec = input_specs[input_name]
                if not isinstance(spec, list) or not spec or not isinstance(spec[0], list):
                    continue
                allowed = spec[0]
                if value in allowed:
                    continue
                replacement = self._pick_combo_replacement(str(value), allowed, preferred_values.get(input_name))
                if replacement is not None:
                    inputs[input_name] = replacement

    @staticmethod
    def _pick_combo_replacement(value: str, allowed: list[Any], preferred: str | None = None) -> Any | None:
        if not allowed:
            return None
        string_allowed = [item for item in allowed if isinstance(item, str)]
        value_lower = value.lower()
        for candidate in string_allowed:
            if candidate.lower() == value_lower:
                return candidate
        if preferred:
            preferred_lower = preferred.lower()
            for candidate in string_allowed:
                if candidate.lower() == preferred_lower:
                    return candidate
        value_stem = Path(value_lower).stem
        for candidate in string_allowed:
            candidate_stem = Path(candidate.lower()).stem
            if value_stem and (value_stem == candidate_stem or value_stem in candidate_stem or candidate_stem in value_stem):
                return candidate
        return allowed[0]

    @staticmethod
    def _normalize_prompt_values(prompt: dict[str, Any]) -> None:
        # Some workflow conversions emit combo values as integer indexes.
        # Normalize known combo fields to stable string options expected by validators.
        analysis_modes = [
            "Drums Only",
            "Full Audio",
            "Vocals Only",
            "Bass Only",
            "Others Audio",
        ]
        for node in prompt.values():
            if node.get("class_type") == "Audio Analysis":
                inputs = node.setdefault("inputs", {})
                val = inputs.get("analysis_mode")
                if isinstance(val, int):
                    idx = max(0, min(val, len(analysis_modes) - 1))
                    inputs["analysis_mode"] = analysis_modes[idx]

    @staticmethod
    def _find_node_ids(prompt: dict[str, Any], class_type: str) -> list[str]:
        return [nid for nid, node in prompt.items() if node.get("class_type") == class_type]

    def _pick_output_node(self, prompt: dict[str, Any]) -> str:
        candidates = []
        for nid, node in prompt.items():
            if node.get("class_type") != "VHS_VideoCombine":
                continue
            candidates.append((nid, node))

        if not candidates:
            raise RuntimeError("No VHS_VideoCombine node found in Yvann template")

        preferred = self.config.yvann_output_node_title.lower().strip()
        for nid, node in candidates:
            meta = json.dumps(node.get("_meta", {})).lower()
            if preferred and preferred in meta:
                return nid

        # Fall back to first candidate for safety.
        return candidates[0][0]

    def _inject_chunk_into_yvann_prompt(self, base_prompt: dict[str, Any], chunk: Chunk) -> tuple[dict[str, Any], str]:
        prompt = copy.deepcopy(base_prompt)

        # Copy chunk inputs into Comfy input folder where loader nodes can find them.
        comfy_input = self.comfy_root / "input"
        comfy_input.mkdir(parents=True, exist_ok=True)

        local_audio = Path(chunk.audio_chunk_path)
        target_audio_name = f"{self.job_id}_{chunk.chunk_id}.wav"
        target_audio = comfy_input / target_audio_name
        shutil.copy2(local_audio, target_audio)

        batch_images = [Path(p) for p in chunk.assigned_image_paths if Path(p).exists()]
        if not batch_images:
            raise RuntimeError(f"No generated images found for {chunk.chunk_id}")
        batch_dir = batch_images[0].parent
        batch_loader_nodes = self._find_node_ids(prompt, "LoadImagesFromFolderKJ")
        if batch_loader_nodes:
            batch_loader_id = batch_loader_nodes[0]
            prompt[batch_loader_id].setdefault("inputs", {}).update(
                {
                    "folder": str(batch_dir),
                    "width": self._vae_safe_dimension(self.config.image_width),
                    "height": self._vae_safe_dimension(self.config.image_height),
                    "keep_aspect_ratio": "crop",
                    "image_load_cap": 0,
                    "start_index": 0,
                    "include_subfolders": False,
                }
            )
        else:
            batch_loader_id = "longform_batch_images"
            prompt[batch_loader_id] = {
                "class_type": "LoadImagesFromFolderKJ",
                "inputs": {
                    "folder": str(batch_dir),
                    "width": self._vae_safe_dimension(self.config.image_width),
                    "height": self._vae_safe_dimension(self.config.image_height),
                    "keep_aspect_ratio": "crop",
                    "image_load_cap": 0,
                    "start_index": 0,
                    "include_subfolders": False,
                },
                "_meta": {"title": f"Longform generated image batch {chunk.visual_batch_id or chunk.chunk_id}"},
            }

        image_batch_nodes = self._find_node_ids(prompt, "ImageBatchMulti")
        original_batch_node = image_batch_nodes[0] if image_batch_nodes else None
        for node in prompt.values():
            inputs = node.setdefault("inputs", {})
            for key, value in list(inputs.items()):
                if original_batch_node and value == [original_batch_node, 0]:
                    inputs[key] = [batch_loader_id, 0]

        load_audio = self._find_node_ids(prompt, "LoadAudio")
        if not load_audio:
            raise RuntimeError("No LoadAudio nodes found in converted Yvann prompt")
        for nid in load_audio:
            prompt[nid].setdefault("inputs", {})["audio"] = target_audio_name

        # Derive practical per-chunk frame settings for long-form processing.
        target_frames = int(round(chunk.chunk_duration * self.config.yvann_render_fps))
        target_frames = max(self.config.yvann_min_frames, min(target_frames, self.config.yvann_max_frames))
        target_width = self._vae_safe_dimension(self.config.image_width)
        target_height = self._vae_safe_dimension(self.config.image_height)
        for nid, node in prompt.items():
            ct = node.get("class_type")
            inputs = node.setdefault("inputs", {})
            title = str(node.get("_meta", {}).get("title", "")).lower()
            if ct == "INTConstant" and "frame" in title and isinstance(inputs.get("value"), int):
                # Workflow group constants use this for animation batch size.
                inputs["value"] = target_frames
            if ct == "INTConstant" and "width" in title and isinstance(inputs.get("value"), int):
                inputs["value"] = target_width
            if ct == "INTConstant" and "height" in title and isinstance(inputs.get("value"), int):
                inputs["value"] = target_height
            if ct == "FloatConstant" and "frame" in title and isinstance(inputs.get("value"), (int, float)):
                # Workflow group constants use this for frame rate.
                inputs["value"] = float(self.config.yvann_render_fps)

        out_node = self._pick_output_node(prompt)
        out_prefix = f"longform_yvann/{self.job_id}/chunks/{chunk.chunk_id}"
        prompt[out_node].setdefault("inputs", {})["filename_prefix"] = out_prefix
        prompt[out_node]["inputs"]["save_output"] = True

        return prompt, out_node

    def render_chunk_video(self, chunk: Chunk, base_prompt: dict[str, Any]) -> None:
        out_path = Path(chunk.video_chunk_path)
        if out_path.exists() and not self.config.overwrite:
            return

        prompt, output_node_id = self._inject_chunk_into_yvann_prompt(base_prompt, chunk)
        prompt_id = self.client.queue_prompt(prompt, partial_targets=[output_node_id])
        history = self.client.wait_for_completion(prompt_id)

        out_info = history.get("outputs", {}).get(output_node_id, {}).get("gifs", [])
        if not out_info:
            # Some VHS versions return "videos" instead of "gifs".
            out_info = history.get("outputs", {}).get(output_node_id, {}).get("videos", [])
        if not out_info:
            raise RuntimeError(f"No video output returned for node {output_node_id}")

        item = out_info[-1]
        src = self.comfy_root / "output" / item.get("subfolder", "") / item["filename"]
        if not src.exists():
            raise FileNotFoundError(f"Expected video output not found: {src}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_path)

    def concat_videos(self, chunks: list[Chunk]) -> Path:
        concat_list = self.final_dir / "concat_list.txt"
        lines = []
        for c in chunks:
            p = Path(c.video_chunk_path)
            if p.exists():
                lines.append(f"file '{p}'")
        if not lines:
            raise RuntimeError("No chunk videos available for concatenation")
        concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

        final_out = self.final_dir / "final_concat.mp4"
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c:v",
                self.config.ffmpeg_video_codec,
                "-crf",
                str(self.config.ffmpeg_crf),
                "-c:a",
                "aac",
                str(final_out),
            ]
        )
        return final_out

    def run(self, dry_run: bool = False) -> dict[str, Any]:
        self.validate()
        chunks, state = self.prepare_job()

        base_prompt: dict[str, Any] | None = None
        if not dry_run:
            base_prompt = self._load_and_convert_yvann_template()

        for chunk in chunks:
            if chunk.status == "completed" and Path(chunk.video_chunk_path).exists() and not self.config.overwrite:
                continue

            state["current_chunk_index"] = chunk.index
            state["updated_at"] = now_utc()
            state["timestamps"]["last_update"] = now_utc()

            try:
                self.split_audio_for_chunk(chunk)
                state["image_generation_status"].setdefault(chunk.chunk_id, "pending")
                self.generate_images_for_chunk(chunk)
                state["image_generation_status"][chunk.chunk_id] = "completed"

                if not dry_run:
                    state["video_generation_status"].setdefault(chunk.chunk_id, "pending")
                    assert base_prompt is not None
                    self.render_chunk_video(chunk, base_prompt)
                    state["video_generation_status"][chunk.chunk_id] = "completed"

                chunk.status = "completed" if not dry_run else "planned"
                chunk.error = None
                if chunk.chunk_id not in state["completed_chunks"]:
                    state["completed_chunks"].append(chunk.chunk_id)
            except Exception as exc:  # noqa: BLE001
                chunk.status = "failed"
                chunk.error = str(exc)
                if chunk.chunk_id not in state["failed_chunks"]:
                    state["failed_chunks"].append(chunk.chunk_id)
                state["video_generation_status"][chunk.chunk_id] = "failed"
                if self.config.stop_on_failure:
                    self._write_manifest(chunks)
                    self._write_job_state(state)
                    raise
            finally:
                self._write_manifest(chunks)
                self._write_job_state(state)

        if not dry_run and self.config.final_concat:
            try:
                self.concat_videos(chunks)
                state["final_concat_status"] = "completed"
            except Exception as exc:  # noqa: BLE001
                state["final_concat_status"] = f"failed: {exc}"
            self._write_job_state(state)

        return {
            "job_id": self.job_id,
            "job_dir": str(self.job_dir),
            "chunk_count": len(chunks),
            "completed": len(state["completed_chunks"]),
            "failed": len(state["failed_chunks"]),
            "dry_run": dry_run,
            "final_concat_status": state.get("final_concat_status"),
        }


def load_config(path: Path) -> JobConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return JobConfig(**data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Long-form Yvann scripted image-to-video orchestrator")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--dry-run", action="store_true", help="Run planning/chunking/image generation but skip Yvann video rendering")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    runner = LongformYvannRunner(cfg)
    result = runner.run(dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
