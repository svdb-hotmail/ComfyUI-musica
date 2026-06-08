#!/usr/bin/env python3
"""Durable longform LTX-2.3 image/audio-to-video orchestration.

This runner is intentionally additive. It prepares shot manifests, patches a
short-form LTX workflow template per shot, queues each shot through ComfyUI's
API, persists state for resume/cancel, and optionally concatenates final clips.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: requests. Install with `pip install requests`.") from exc

try:
    from longform_yvann_cue_parser import parse_visual_cue_markers
except ImportError:  # pragma: no cover
    from script_examples.longform_yvann_cue_parser import parse_visual_cue_markers


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sec_to_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000.0))
    return f"{whole // 3600:02d}:{(whole % 3600) // 60:02d}:{whole % 60:02d}.{ms:03d}"


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


class JobCancelled(RuntimeError):
    pass


@dataclasses.dataclass
class LTXShot:
    shot_id: str
    index: int
    start_time: float
    end_time: float
    duration: float
    prompt: str
    summary: str
    image_path: str
    audio_chunk_path: str
    video_path: str
    seed: int
    status: str = "planned"
    error: str | None = None
    prompt_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class LTXJobConfig:
    audio_path: str
    image_paths: list[str]
    prompt_plan_path: str
    output_root: str

    comfy_api_url: str = "http://127.0.0.1:18188"
    comfy_api_verify_tls: bool = False
    comfy_root: str = "."
    workflow_template_path: str = "script_examples/workflows/Movie_Builder_LTX2.3_workflow.json"
    renderer: str = "movie_builder"  # movie_builder | ia2v

    global_style_prompt: str = (
        "cinematic music video, coherent image-to-video motion, strong subject continuity, "
        "rhythmic visual evolution, polished lighting, no readable text, no watermark"
    )
    negative_prompt: str = "text, subtitles, logo, watermark, still image, frozen video, blurry, low quality, distorted"
    shot_duration_seconds: float = 6.0
    max_shots: int | None = None
    width: int = 1280
    height: int = 720
    fps: int = 24
    seed_strategy: str = "derived"  # derived | deterministic | random
    base_seed: int = 42
    seed_offset: int = 1009
    use_previous_final_frame: bool = True
    prompt_enhance: bool = True
    enable_upscale: bool = False
    enable_voice_reference: bool = False
    resume: bool = True
    overwrite: bool = False
    stop_on_failure: bool = False
    final_concat: bool = True
    ffmpeg_crf: int = 18
    job_id: str | None = None
    resume_job_dir: str | None = None

    def __post_init__(self) -> None:
        self.renderer = str(self.renderer or "movie_builder").strip().lower()
        if self.renderer not in {"movie_builder", "ia2v"}:
            raise ValueError("renderer must be 'movie_builder' or 'ia2v'")
        self.shot_duration_seconds = max(1.0, float(self.shot_duration_seconds))
        self.width = int(self.width)
        self.height = int(self.height)
        self.fps = int(self.fps)


class ComfyClient:
    def __init__(self, base_url: str, verify_tls: bool = False):
        self.base_url = base_url.rstrip("/")
        self.verify_tls = verify_tls
        self.session = requests.Session()

    def get_json(self, path: str) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}{path}", timeout=30, verify=self.verify_tls)
        response.raise_for_status()
        return response.json()

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(f"{self.base_url}{path}", json=payload, timeout=120, verify=self.verify_tls)
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code} on {path}: {response.text[:4000]}")
        return response.json()

    def post(self, path: str, payload: dict[str, Any] | None = None) -> None:
        response = self.session.post(f"{self.base_url}{path}", json=payload or {}, timeout=30, verify=self.verify_tls)
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code} on {path}: {response.text[:4000]}")

    def convert_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        return self.post_json("/workflow/convert", workflow)

    def queue_prompt(self, prompt: dict[str, Any], partial_targets: list[str] | None = None) -> str:
        payload: dict[str, Any] = {"prompt": prompt}
        if partial_targets:
            payload["partial_execution_targets"] = partial_targets
        return str(self.post_json("/prompt", payload)["prompt_id"])

    def wait_for_completion(self, prompt_id: str, cancel_requested: Any | None = None, poll_seconds: float = 2.0) -> dict[str, Any]:
        while True:
            if cancel_requested is not None and cancel_requested():
                try:
                    self.post("/interrupt", {})
                finally:
                    raise JobCancelled(f"Job cancelled while waiting for prompt {prompt_id}")
            history = self.get_json(f"/history/{prompt_id}")
            if history and prompt_id in history:
                item = history[prompt_id]
                if item.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"Prompt failed: {item.get('node_errors', {})}")
                return item
            time.sleep(poll_seconds)


class LongformLTX23Runner:
    def __init__(self, config: LTXJobConfig):
        self.config = config
        self.comfy_root = Path(config.comfy_root).resolve()
        self.audio_path = self._resolve_path(config.audio_path)
        self.prompt_plan_path = self._resolve_path(config.prompt_plan_path)
        self.workflow_template_path = self._resolve_path(config.workflow_template_path)
        self.image_paths = [self._resolve_path(path) for path in config.image_paths]
        self.output_root = self._resolve_path(config.output_root)
        self.job_id = config.job_id or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.job_dir = self.output_root / self.job_id
        self.manifest_dir = self.job_dir / "manifest"
        self.audio_chunks_dir = self.job_dir / "audio_chunks"
        self.videos_dir = self.job_dir / "videos"
        self.frames_dir = self.job_dir / "frames"
        self.final_dir = self.job_dir / "final"
        self.state_path = self.job_dir / "job_state.json"
        self.manifest_path = self.manifest_dir / "shot_manifest.json"
        self.cancel_path = self.job_dir / "cancel.requested"
        self.client = ComfyClient(config.comfy_api_url, config.comfy_api_verify_tls)
        self.audio_duration: float | None = None

    def _resolve_path(self, value: str) -> Path:
        path = Path(str(value).strip())
        if path.is_absolute():
            return path.resolve()
        return (self.comfy_root / path).resolve()

    def validate(self, check_api: bool = True) -> None:
        if not self.audio_path.exists():
            raise FileNotFoundError(f"audio_path does not exist: {self.audio_path}")
        if not self.prompt_plan_path.exists():
            raise FileNotFoundError(f"prompt_plan_path does not exist: {self.prompt_plan_path}")
        if not self.workflow_template_path.exists():
            raise FileNotFoundError(f"workflow_template_path does not exist: {self.workflow_template_path}")
        if not self.image_paths:
            raise ValueError("At least one input image is required")
        for image_path in self.image_paths:
            if not image_path.exists():
                raise FileNotFoundError(f"image_path does not exist: {image_path}")
        run_cmd(["ffmpeg", "-version"])
        run_cmd(["ffprobe", "-version"])
        self.audio_duration = self.get_audio_duration(self.audio_path)
        if self.audio_duration <= 1.0:
            raise ValueError("Audio duration looks invalid")
        if self.audio_duration > 10 * 60 + 1:
            raise ValueError("This MVP is capped at soundtracks of 10 minutes or less")
        self.job_dir.mkdir(parents=True, exist_ok=True)
        if check_api:
            self.client.get_json("/system_stats")

    @staticmethod
    def get_audio_duration(audio_path: Path) -> float:
        proc = run_cmd([
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ])
        return float(proc.stdout.strip())

    def load_plan_text(self) -> str:
        return self.prompt_plan_path.read_text(encoding="utf-8", errors="replace")

    def _extract_cues(self, total_duration: float) -> list[dict[str, Any]]:
        cues = parse_visual_cue_markers(self.load_plan_text(), total_duration)
        return [cue for cue in cues if float(cue["end"]) > float(cue["start"])]

    def _shot_boundaries(self, total_duration: float, cues: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
        if cues:
            boundaries = [(float(cue["start"]), float(cue["end"]), str(cue["summary"])) for cue in cues]
        else:
            boundaries = []
            t = 0.0
            while t < total_duration:
                end = min(total_duration, t + self.config.shot_duration_seconds)
                boundaries.append((t, end, self.load_plan_text().strip() or "Audio-reactive cinematic music video shot."))
                t = end
        if self.config.max_shots is not None:
            boundaries = boundaries[: int(self.config.max_shots)]
        return boundaries

    def _seed_for(self, shot_index: int, prompt: str) -> int:
        if self.config.seed_strategy == "random":
            return random.randint(0, 2_147_483_647)
        if self.config.seed_strategy == "deterministic":
            return self.config.base_seed + shot_index * self.config.seed_offset
        digest = hashlib.sha256(f"{self.config.base_seed}|{shot_index}|{prompt}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _build_prompt(self, summary: str, start: float, end: float, previous: str | None) -> str:
        continuity = ""
        if previous:
            continuity = f" Keep continuity from the previous shot: {previous[:220]}."
        return (
            f"{self.config.global_style_prompt}. "
            f"Shot {sec_to_hms(start)} to {sec_to_hms(end)}: {summary.strip()} "
            "Describe a complete moving scene with subject identity, action over time, camera movement, lighting, atmosphere, "
            "audio-reactive motion, and a clear final frame. "
            f"Audio for this window drives the motion and rhythm.{continuity} "
            f"Negative instructions: {self.config.negative_prompt}."
        )

    def build_manifest(self) -> list[LTXShot]:
        total_duration = self.audio_duration or self.get_audio_duration(self.audio_path)
        cues = self._extract_cues(total_duration)
        boundaries = self._shot_boundaries(total_duration, cues)
        shots: list[LTXShot] = []
        previous_summary: str | None = None
        for index, (start, end, summary) in enumerate(boundaries, start=1):
            image_path = self.image_paths[(index - 1) % len(self.image_paths)]
            shot_id = f"shot_{index:04d}"
            prompt = self._build_prompt(summary, start, end, previous_summary)
            seed = self._seed_for(index, prompt)
            shots.append(
                LTXShot(
                    shot_id=shot_id,
                    index=index,
                    start_time=start,
                    end_time=end,
                    duration=end - start,
                    prompt=prompt,
                    summary=summary.strip(),
                    image_path=str(image_path),
                    audio_chunk_path=str(self.audio_chunks_dir / f"{shot_id}.wav"),
                    video_path=str(self.videos_dir / f"{shot_id}.mp4"),
                    seed=seed,
                )
            )
            previous_summary = summary.strip()
        return shots

    def _write_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_path, state)

    def _write_manifest(self, shots: list[LTXShot]) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "job_id": self.job_id,
                "created_at": now_utc(),
                "renderer": self.config.renderer,
                "source_audio_path": str(self.audio_path),
                "source_prompt_plan_path": str(self.prompt_plan_path),
                "workflow_template_path": str(self.workflow_template_path),
                "shots": [shot.as_dict() for shot in shots],
            },
        )

    def _initial_state(self, shots: list[LTXShot]) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "created_at": now_utc(),
            "updated_at": now_utc(),
            "status": "running",
            "renderer": self.config.renderer,
            "number_of_shots": len(shots),
            "current_shot_index": 0,
            "current_shot_id": "",
            "current_stage": "starting",
            "completed_shots": [],
            "failed_shots": [],
            "final_concat_status": "pending",
            "cancel_requested": False,
        }

    def _cancel_requested(self) -> bool:
        return self.cancel_path.exists()

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested():
            raise JobCancelled("Job cancellation requested")

    def split_audio_for_shot(self, shot: LTXShot) -> None:
        out = Path(shot.audio_chunk_path)
        if out.exists() and not self.config.overwrite:
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        run_cmd([
            "ffmpeg",
            "-y",
            "-ss",
            f"{shot.start_time:.3f}",
            "-t",
            f"{shot.duration:.3f}",
            "-i",
            str(self.audio_path),
            "-ac",
            "2",
            "-ar",
            "48000",
            str(out),
        ])

    @staticmethod
    def _workflow_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = list(workflow.get("nodes", []))
        for subgraph in (workflow.get("definitions") or {}).get("subgraphs", []):
            nodes.extend(subgraph.get("nodes", []))
        return nodes

    @staticmethod
    def _set_widget(node: dict[str, Any], value: Any, index: int = 0) -> bool:
        widgets = node.setdefault("widgets_values", [])
        if not isinstance(widgets, list):
            return False
        while len(widgets) <= index:
            widgets.append(None)
        widgets[index] = value
        return True

    @staticmethod
    def _find_nodes(workflow: dict[str, Any], node_type: str | None = None, title: str | None = None) -> list[dict[str, Any]]:
        matches = []
        for node in LongformLTX23Runner._workflow_nodes(workflow):
            if node_type and node.get("type") != node_type:
                continue
            if title and str(node.get("title") or "") != title:
                continue
            matches.append(node)
        return matches

    def _patch_common_loaders(self, workflow: dict[str, Any], shot: LTXShot) -> None:
        image_name = Path(shot.image_path).name
        audio_name = Path(shot.audio_chunk_path).name
        for node in self._find_nodes(workflow, "LoadImage"):
            self._set_widget(node, image_name, 0)
        for node in self._find_nodes(workflow, "LoadAudio"):
            self._set_widget(node, audio_name, 0)
        for node in self._find_nodes(workflow, "SaveVideo"):
            prefix = f"longform_ltx23/{self.job_id}/{shot.shot_id}"
            self._set_widget(node, prefix, 0)

    def _patch_ia2v_workflow(self, workflow: dict[str, Any], shot: LTXShot) -> dict[str, Any]:
        self._patch_common_loaders(workflow, shot)
        by_id = {str(node.get("id")): node for node in self._workflow_nodes(workflow)}
        for node_id, value in {
            "319": shot.prompt,
            "349": bool(self.config.prompt_enhance),
            "330": int(self.config.width),
            "324": int(self.config.height),
            "332": 0.0,
            "331": float(shot.duration),
            "323": int(self.config.fps),
            "286": int(shot.seed),
        }.items():
            node = by_id.get(node_id)
            if node:
                self._set_widget(node, value, 0)
        return workflow

    def _patch_movie_builder_workflow(self, workflow: dict[str, Any], shot: LTXShot) -> dict[str, Any]:
        self._patch_common_loaders(workflow, shot)
        for node in self._find_nodes(workflow, "PrimitiveStringMultiline", "Text Prompt"):
            self._set_widget(node, shot.prompt, 0)
        for node in self._find_nodes(workflow, "TrimAudioDuration"):
            self._set_widget(node, 0, 0)
            self._set_widget(node, float(shot.duration), 1)
        for node in self._find_nodes(workflow, "PrimitiveInt", "Frame Rate"):
            self._set_widget(node, int(self.config.fps), 0)
        for node in self._find_nodes(workflow, "PrimitiveInt", "Length"):
            frame_count = max(1, int(round(shot.duration * self.config.fps)) + 1)
            self._set_widget(node, frame_count, 0)
        for node in self._find_nodes(workflow, "LtxResolutionPicker"):
            widgets = node.setdefault("widgets_values", [])
            if isinstance(widgets, list) and len(widgets) >= 3:
                widgets[2] = int(self.config.width)
                widgets[3] = int(self.config.height)
        for node in self._find_nodes(workflow, "RandomNoise"):
            self._set_widget(node, int(shot.seed), 0)
        for node in self._find_nodes(workflow, "ShotVideoOutput"):
            self._set_widget(node, int(shot.index), 0)
        for node in self._find_nodes(workflow, "PrimitiveBoolean", "Enable Upscale"):
            self._set_widget(node, bool(self.config.enable_upscale), 0)
        for node in self._find_nodes(workflow, "PrimitiveBoolean", "Enable Voice Reference"):
            self._set_widget(node, bool(self.config.enable_voice_reference), 0)
        return workflow

    def patched_workflow_for_shot(self, shot: LTXShot) -> dict[str, Any]:
        workflow = json.loads(self.workflow_template_path.read_text(encoding="utf-8"))
        if self.config.renderer == "ia2v":
            return self._patch_ia2v_workflow(workflow, shot)
        return self._patch_movie_builder_workflow(workflow, shot)

    def _copy_input_for_comfy_loader(self, path: Path) -> None:
        input_dir = self.comfy_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        dst = input_dir / path.name
        if dst.exists() and dst.resolve() == path.resolve():
            return
        if not dst.exists() or self.config.overwrite:
            shutil.copy2(path, dst)

    def _output_video_from_history(self, history: dict[str, Any]) -> Path | None:
        outputs = history.get("outputs", {})
        for output in outputs.values():
            for key in ("videos", "gifs"):
                for item in output.get(key, []) if isinstance(output, dict) else []:
                    filename = item.get("filename")
                    if not filename:
                        continue
                    subfolder = item.get("subfolder") or ""
                    kind = item.get("type") or "output"
                    root = self.comfy_root / kind if kind in {"input", "output", "temp"} else self.comfy_root / "output"
                    candidate = root / subfolder / filename
                    if candidate.exists():
                        return candidate
        return None

    def render_shot(self, shot: LTXShot) -> None:
        out_path = Path(shot.video_path)
        if out_path.exists() and not self.config.overwrite:
            return
        self._copy_input_for_comfy_loader(Path(shot.image_path))
        self._copy_input_for_comfy_loader(Path(shot.audio_chunk_path))
        workflow = self.patched_workflow_for_shot(shot)
        api_prompt = self.client.convert_workflow(workflow)
        prompt_id = self.client.queue_prompt(api_prompt)
        shot.prompt_id = prompt_id
        history = self.client.wait_for_completion(prompt_id, cancel_requested=self._cancel_requested)
        generated = self._output_video_from_history(history)
        if generated is None:
            raise RuntimeError("No video output found in Comfy history")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, out_path)

    def concat_videos(self, shots: list[LTXShot]) -> Path:
        final = self.final_dir / "final_concat.mp4"
        final.parent.mkdir(parents=True, exist_ok=True)
        list_file = self.final_dir / "concat_list.txt"
        with list_file.open("w", encoding="utf-8") as handle:
            for shot in shots:
                path = Path(shot.video_path)
                if path.exists():
                    handle.write(f"file '{path.as_posix()}'\n")
        run_cmd([
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-i",
            str(self.audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-crf",
            str(self.config.ffmpeg_crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(final),
        ])
        return final

    def prepare_job(self) -> tuple[list[LTXShot], dict[str, Any]]:
        for directory in [self.manifest_dir, self.audio_chunks_dir, self.videos_dir, self.frames_dir, self.final_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        shots = self.build_manifest()
        self._write_manifest(shots)
        state = self._initial_state(shots)
        self._write_state(state)
        shutil.copy2(self.audio_path, self.job_dir / self.audio_path.name)
        self.job_dir.joinpath("prompt_plan_source.txt").write_text(self.load_plan_text(), encoding="utf-8")
        return shots, state

    def run(self, dry_run: bool = False) -> dict[str, Any]:
        self.validate(check_api=not dry_run)
        shots, state = self.prepare_job()
        if dry_run:
            state["status"] = "planned"
            state["current_stage"] = "planned"
            self._write_state(state)
            return {"job_id": self.job_id, "job_dir": str(self.job_dir), "shot_count": len(shots), "dry_run": True}

        for shot in shots:
            if self._cancel_requested():
                state["status"] = "cancelled"
                state["cancel_requested"] = True
                self._write_state(state)
                break
            if Path(shot.video_path).exists() and not self.config.overwrite:
                if shot.shot_id not in state["completed_shots"]:
                    state["completed_shots"].append(shot.shot_id)
                continue
            state["current_shot_index"] = shot.index
            state["current_shot_id"] = shot.shot_id
            state["current_stage"] = "splitting_audio"
            state["updated_at"] = now_utc()
            self._write_state(state)
            try:
                self._raise_if_cancelled()
                self.split_audio_for_shot(shot)
                state["current_stage"] = "rendering_ltx_video"
                state["updated_at"] = now_utc()
                self._write_state(state)
                self.render_shot(shot)
                shot.status = "completed"
                if shot.shot_id not in state["completed_shots"]:
                    state["completed_shots"].append(shot.shot_id)
            except JobCancelled as exc:
                shot.status = "cancelled"
                shot.error = str(exc)
                state["status"] = "cancelled"
                state["cancel_requested"] = True
                self._write_manifest(shots)
                self._write_state(state)
                break
            except Exception as exc:  # noqa: BLE001
                shot.status = "failed"
                shot.error = str(exc)
                if shot.shot_id not in state["failed_shots"]:
                    state["failed_shots"].append(shot.shot_id)
                self._write_manifest(shots)
                self._write_state(state)
                if self.config.stop_on_failure:
                    raise
            finally:
                self._write_manifest(shots)
                self._write_state(state)

        if state.get("status") != "cancelled" and self.config.final_concat:
            try:
                state["current_stage"] = "concatenating_final_video"
                self._write_state(state)
                self.concat_videos(shots)
                state["final_concat_status"] = "completed"
                state["status"] = "completed" if not state["failed_shots"] else "failed"
            except Exception as exc:  # noqa: BLE001
                state["final_concat_status"] = f"failed: {exc}"
                state["status"] = "failed"
            self._write_state(state)
        elif state.get("status") != "cancelled":
            state["status"] = "completed" if not state["failed_shots"] else "failed"
            self._write_state(state)

        return {
            "job_id": self.job_id,
            "job_dir": str(self.job_dir),
            "shot_count": len(shots),
            "completed": len(state["completed_shots"]),
            "failed": len(state["failed_shots"]),
            "final_concat_status": state.get("final_concat_status"),
        }


def load_config(path: Path) -> LTXJobConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return LTXJobConfig(**data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run durable longform LTX-2.3 shot rendering")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = LongformLTX23Runner(load_config(args.config)).run(dry_run=args.dry_run)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
