#!/usr/bin/env python3
"""Run an ordered LTX 2.3 first-frame/last-frame keyframe plan."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import re
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


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def parse_timecode(value: str) -> float:
    parts = [int(part) for part in value.strip().split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    raise ValueError(f"Unsupported timecode: {value}")


@dataclasses.dataclass
class FLF2VClip:
    clip_id: str
    index: int
    title: str
    start_time: float
    end_time: float
    duration: int
    prompt: str
    first_frame_path: str
    last_frame_path: str
    first_frame_loader_name: str
    last_frame_loader_name: str
    video_path: str
    seed: int
    prompt_id: str | None = None
    status: str = "planned"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class FLF2VJobConfig:
    prompt_plan_path: str
    keyframe_dir: str
    output_root: str
    workflow_template_path: str = "user/default/workflows/video_ltx2_3_flf2v.json"
    audio_path: str | None = None
    comfy_root: str = "."
    comfy_api_url: str = "http://127.0.0.1:18188"
    comfy_api_verify_tls: bool = False
    job_id: str | None = None
    width: int = 1280
    height: int = 720
    fps: int = 24
    seed_strategy: str = "random"
    base_seed: int = 42
    seed_offset: int = 1009
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    include_plan_global_context: bool = True
    include_plan_negative_prompt: bool = True
    final_concat: bool = True
    overwrite: bool = False
    stop_on_failure: bool = False
    max_clips: int | None = None

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.fps = int(self.fps)
        self.seed_strategy = str(self.seed_strategy or "random").strip().lower()
        if self.seed_strategy not in {"random", "deterministic", "derived"}:
            raise ValueError("seed_strategy must be random, deterministic, or derived")


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

    def convert_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        return self.post_json("/workflow/convert", workflow)

    def queue_prompt(self, prompt: dict[str, Any]) -> str:
        return str(self.post_json("/prompt", {"prompt": prompt})["prompt_id"])

    def wait_for_completion(self, prompt_id: str, poll_seconds: float = 2.0) -> dict[str, Any]:
        while True:
            history = self.get_json(f"/history/{prompt_id}")
            if history and prompt_id in history:
                item = history[prompt_id]
                if item.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"Prompt failed: {item.get('node_errors', {})}")
                return item
            time.sleep(poll_seconds)


class FLF2VRunner:
    def __init__(self, config: FLF2VJobConfig):
        self.config = config
        self.comfy_root = Path(config.comfy_root).resolve()
        self.prompt_plan_path = self._resolve(config.prompt_plan_path)
        self.keyframe_dir = self._resolve(config.keyframe_dir)
        self.workflow_template_path = self._resolve(config.workflow_template_path)
        self.output_root = self._resolve(config.output_root)
        self.audio_path = self._resolve(config.audio_path) if config.audio_path else None
        self.job_id = config.job_id or f"flf2v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.job_dir = self.output_root / self.job_id
        self.manifest_dir = self.job_dir / "manifest"
        self.videos_dir = self.job_dir / "videos"
        self.final_dir = self.job_dir / "final"
        self.input_copy_dir = self.comfy_root / "input" / f"{self.job_id}_keyframes"
        self.state_path = self.job_dir / "job_state.json"
        self.manifest_path = self.manifest_dir / "clip_manifest.json"
        self.client = ComfyClient(config.comfy_api_url, config.comfy_api_verify_tls)

    def _resolve(self, value: str | None) -> Path:
        if value is None:
            raise ValueError("Cannot resolve None path")
        path = Path(str(value).strip())
        if path.is_absolute():
            return path.resolve()
        return (self.comfy_root / path).resolve()

    def validate(self, check_api: bool = True) -> None:
        for path, label in [
            (self.prompt_plan_path, "prompt_plan_path"),
            (self.keyframe_dir, "keyframe_dir"),
            (self.workflow_template_path, "workflow_template_path"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if self.audio_path and not self.audio_path.exists():
            raise FileNotFoundError(f"audio_path does not exist: {self.audio_path}")
        run_cmd(["ffmpeg", "-version"])
        run_cmd(["ffprobe", "-version"])
        if check_api:
            self.client.get_json("/system_stats")

    @staticmethod
    def _keyframe_index(path: Path) -> int:
        match = re.search(r"\((\d+)\)", path.stem)
        if match:
            return int(match.group(1))
        digits = re.findall(r"\d+", path.stem)
        return int(digits[-1]) if digits else -1

    def keyframes(self) -> list[Path]:
        paths = [p for p in self.keyframe_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        return sorted(paths, key=lambda path: (self._keyframe_index(path), path.name.lower()))

    def parse_plan(self) -> list[dict[str, Any]]:
        text = self.prompt_plan_path.read_text(encoding="utf-8", errors="replace")
        pattern = re.compile(
            r"^##\s+Clip\s+(?P<label>.+?)\s+[\u2014-]\s+(?P<start>\d{2}:\d{2})[\u2013-](?P<end>\d{2}:\d{2})\s+[\u2014-]\s+(?P<title>.+?)\s*$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        clips: list[dict[str, Any]] = []
        for index, match in enumerate(matches, start=1):
            block_start = match.end()
            block_end = matches[index].start() if index < len(matches) else len(text)
            block = text[block_start:block_end]
            prompt_match = re.search(r"###\s+FLF2V Prompt\s*\n(?P<prompt>.*?)(?:\n---|\Z)", block, re.DOTALL)
            if not prompt_match:
                raise ValueError(f"Missing FLF2V Prompt for clip {match.group('label')}")
            start = parse_timecode(match.group("start"))
            end = parse_timecode(match.group("end"))
            prompt = prompt_match.group("prompt").strip()
            clips.append(
                {
                    "label": match.group("label").strip(),
                    "title": match.group("title").strip(),
                    "start": start,
                    "end": end,
                    "duration": max(1, int(round(end - start))),
                    "prompt": prompt,
                }
            )
        if self.config.max_clips is not None:
            clips = clips[: int(self.config.max_clips)]
        return clips

    def plan_global_context(self) -> str:
        text = self.prompt_plan_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"##\s+Export Notes\s*(?P<body>.*?)(?:\n---\s*\n\s*#\s+Global Negative Prompt|\Z)", text, re.DOTALL)
        if not match:
            return ""
        lines: list[str] = []
        for line in match.group("body").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith("- total video length") or lower.startswith("- format:") or lower.startswith("- workflow:"):
                continue
            lines.append(stripped)
        if not lines:
            return ""
        return "Global continuity rules from the plan:\n" + "\n".join(lines)

    def plan_negative_prompt(self) -> str:
        text = self.prompt_plan_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"#\s+Global Negative Prompt\s*(?P<body>.*?)(?:\n---\s*\n\s*#\s+Shot List|\Z)", text, re.DOTALL)
        return match.group("body").strip() if match else ""

    def seed_for(self, index: int, prompt: str) -> int:
        if self.config.seed_strategy == "random":
            return random.randint(0, 2_147_483_647)
        if self.config.seed_strategy == "deterministic":
            return self.config.base_seed + index * self.config.seed_offset
        digest = hashlib.sha256(f"{self.config.base_seed}|{index}|{prompt}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def copy_keyframe_for_loader(self, source: Path, index: int) -> str:
        self.input_copy_dir.mkdir(parents=True, exist_ok=True)
        dst = self.input_copy_dir / f"kf_{index:04d}{source.suffix.lower()}"
        if not dst.exists() or self.config.overwrite:
            shutil.copy2(source, dst)
        return f"{self.input_copy_dir.name}/{dst.name}"

    def build_manifest(self) -> list[FLF2VClip]:
        clips = self.parse_plan()
        plan_context = self.plan_global_context() if self.config.include_plan_global_context else ""
        keyframes = self.keyframes()
        if len(keyframes) != len(clips) + 1:
            raise ValueError(f"Expected {len(clips) + 1} keyframes for {len(clips)} FLF2V clips, found {len(keyframes)}")
        shots: list[FLF2VClip] = []
        for index, clip in enumerate(clips, start=1):
            first_frame = keyframes[index - 1]
            last_frame = keyframes[index]
            prompt = "\n\n".join(part for part in [plan_context, self.config.prompt_prefix.strip(), clip["prompt"], self.config.prompt_suffix.strip()] if part)
            shot_id = f"clip_{index:04d}_{clip['label'].lower().replace(' ', '_')}"
            shots.append(
                FLF2VClip(
                    clip_id=shot_id,
                    index=index,
                    title=f"Clip {clip['label']} - {clip['title']}",
                    start_time=float(clip["start"]),
                    end_time=float(clip["end"]),
                    duration=int(clip["duration"]),
                    prompt=prompt,
                    first_frame_path=str(first_frame),
                    last_frame_path=str(last_frame),
                    first_frame_loader_name=self.copy_keyframe_for_loader(first_frame, index - 1),
                    last_frame_loader_name=self.copy_keyframe_for_loader(last_frame, index),
                    video_path=str(self.videos_dir / f"{index:04d}_{clip['label'].lower()}.mp4"),
                    seed=self.seed_for(index, prompt),
                )
            )
        return shots

    @staticmethod
    def workflow_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = list(workflow.get("nodes", []))
        for subgraph in (workflow.get("definitions") or {}).get("subgraphs", []):
            nodes.extend(subgraph.get("nodes", []))
        return nodes

    @staticmethod
    def set_widget(node: dict[str, Any], value: Any, index: int = 0) -> None:
        widgets = node.setdefault("widgets_values", [])
        if not isinstance(widgets, list):
            node["widgets_values"] = []
            widgets = node["widgets_values"]
        while len(widgets) <= index:
            widgets.append(None)
        widgets[index] = value

    def patch_workflow(self, clip: FLF2VClip) -> dict[str, Any]:
        workflow = json.loads(self.workflow_template_path.read_text(encoding="utf-8"))
        nodes = self.workflow_nodes(workflow)
        by_id = {str(node.get("id")): node for node in nodes}
        for node in nodes:
            if node.get("type") == "LoadImage" and node.get("title") == "Load First Frame":
                self.set_widget(node, clip.first_frame_loader_name, 0)
            if node.get("type") == "LoadImage" and node.get("title") == "Load Last Frame":
                self.set_widget(node, clip.last_frame_loader_name, 0)
            if node.get("type") == "SaveVideo":
                self.set_widget(node, f"longform_ltx23/{self.job_id}/{clip.index:04d}_{clip.clip_id}", 0)
        values = {
            "128": clip.prompt,
            "113": int(self.config.width),
            "98": int(self.config.height),
            "102": int(clip.duration),
            "114": int(self.config.fps),
            "100": int(clip.seed),
        }
        for node_id, value in values.items():
            node = by_id.get(node_id)
            if node:
                self.set_widget(node, value, 0)
        negative_node = by_id.get("112")
        if negative_node and self.config.include_plan_negative_prompt:
            widgets = negative_node.get("widgets_values")
            existing_negative = widgets[0] if isinstance(widgets, list) and widgets else ""
            negative_parts = [self.plan_negative_prompt(), str(existing_negative).strip()]
            self.set_widget(negative_node, "\n\n".join(part for part in negative_parts if part), 0)
        return workflow

    def output_video_from_history(self, history: dict[str, Any]) -> Path | None:
        for output in history.get("outputs", {}).values():
            if not isinstance(output, dict):
                continue
            for key in ("videos", "gifs", "images"):
                for item in output.get(key, []):
                    filename = item.get("filename")
                    if not filename or Path(str(filename)).suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv", ".gif"}:
                        continue
                    subfolder = item.get("subfolder") or ""
                    kind = item.get("type") or "output"
                    root = self.comfy_root / kind if kind in {"input", "output", "temp"} else self.comfy_root / "output"
                    candidate = root / subfolder / filename
                    if candidate.exists():
                        return candidate
        return None

    def render_clip(self, clip: FLF2VClip) -> None:
        out = Path(clip.video_path)
        if out.exists() and not self.config.overwrite:
            return
        workflow = self.patch_workflow(clip)
        api_prompt = self.client.convert_workflow(workflow)
        prompt_id = self.client.queue_prompt(api_prompt)
        clip.prompt_id = prompt_id
        history = self.client.wait_for_completion(prompt_id)
        generated = self.output_video_from_history(history)
        if generated is None:
            raise RuntimeError("No video output found in Comfy history")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, out)

    def concat_videos(self, clips: list[FLF2VClip]) -> Path:
        self.final_dir.mkdir(parents=True, exist_ok=True)
        list_file = self.final_dir / "concat_list.txt"
        with list_file.open("w", encoding="utf-8") as handle:
            for clip in clips:
                path = Path(clip.video_path)
                if path.exists():
                    handle.write(f"file '{path.as_posix()}'\n")
        concat_path = self.final_dir / "final_flf2v_concat.mp4"
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(concat_path)])
        if self.audio_path:
            remuxed = self.final_dir / "final_flf2v_continuous_audio.mp4"
            run_cmd([
                "ffmpeg", "-y", "-i", str(concat_path), "-i", str(self.audio_path),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(remuxed),
            ])
            return remuxed
        return concat_path

    def write_manifest(self, clips: list[FLF2VClip]) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "job_id": self.job_id,
                "updated_at": now_utc(),
                "workflow_template_path": str(self.workflow_template_path),
                "prompt_plan_path": str(self.prompt_plan_path),
                "keyframe_dir": str(self.keyframe_dir),
                "mode": "first_frame_last_frame",
                "clip_count": len(clips),
                "plan_global_context": self.plan_global_context() if self.config.include_plan_global_context else "",
                "plan_negative_prompt": self.plan_negative_prompt() if self.config.include_plan_negative_prompt else "",
                "clips": [clip.as_dict() for clip in clips],
            },
        )

    def run(self, dry_run: bool = False) -> dict[str, Any]:
        self.validate(check_api=not dry_run)
        for directory in [self.job_dir, self.manifest_dir, self.videos_dir, self.final_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        clips = self.build_manifest()
        self.write_manifest(clips)
        self.job_dir.joinpath("prompt_plan_source.txt").write_text(
            self.prompt_plan_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
        )
        state: dict[str, Any] = {
            "job_id": self.job_id,
            "created_at": now_utc(),
            "updated_at": now_utc(),
            "status": "planned" if dry_run else "running",
            "mode": "flf2v_first_frame_last_frame",
            "number_of_clips": len(clips),
            "completed_clips": [],
            "failed_clips": [],
            "final_concat_status": "pending",
        }
        atomic_write_json(self.state_path, state)
        if dry_run:
            return {"job_id": self.job_id, "job_dir": str(self.job_dir), "clip_count": len(clips), "dry_run": True}
        for clip in clips:
            state["current_clip_index"] = clip.index
            state["current_clip_id"] = clip.clip_id
            state["updated_at"] = now_utc()
            atomic_write_json(self.state_path, state)
            try:
                self.render_clip(clip)
                clip.status = "completed"
                state["completed_clips"].append(clip.clip_id)
            except Exception as exc:  # noqa: BLE001
                clip.status = "failed"
                clip.error = str(exc)
                state["failed_clips"].append(clip.clip_id)
                self.write_manifest(clips)
                atomic_write_json(self.state_path, state)
                if self.config.stop_on_failure:
                    raise
            self.write_manifest(clips)
            atomic_write_json(self.state_path, state)
        final_path: Path | None = None
        if self.config.final_concat:
            try:
                final_path = self.concat_videos(clips)
                state["final_concat_status"] = "completed"
            except Exception as exc:  # noqa: BLE001
                state["final_concat_status"] = f"failed: {exc}"
                state["status"] = "failed"
        if state.get("status") != "failed":
            state["status"] = "completed" if not state["failed_clips"] else "failed"
        state["updated_at"] = now_utc()
        atomic_write_json(self.state_path, state)
        return {
            "job_id": self.job_id,
            "job_dir": str(self.job_dir),
            "clip_count": len(clips),
            "completed": len(state["completed_clips"]),
            "failed": len(state["failed_clips"]),
            "final_path": str(final_path) if final_path else None,
            "final_concat_status": state.get("final_concat_status"),
        }


def load_config(path: Path) -> FLF2VJobConfig:
    return FLF2VJobConfig(**json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ordered LTX 2.3 FLF2V keyframe clips")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = FLF2VRunner(load_config(args.config)).run(dry_run=args.dry_run)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())