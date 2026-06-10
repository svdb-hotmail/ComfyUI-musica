#!/usr/bin/env python3
"""Generate ordered keyframes from RWBT-style markdown plans with optional AI correction loops.

This runner is designed for plans that contain sections like:
  ## Clip 1A - 00:00-00:20 - Broadcast Begins
  ### START keyframe image prompt
  ```text
  ...
  ```
  ### END keyframe image prompt
  ```text
  ...
  ```

The script:
1) Parses START/END prompts for each clip.
2) Patches a ComfyUI workflow template per task.
3) Queues prompts through Comfy API (/workflow/convert + /prompt).
4) Optionally uses an OpenAI-compatible LLM/VLM endpoint to:
   - interpret and tune prompt/parameters before generation,
   - analyze outputs for continuity issues,
   - propose auto-corrections and retry.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
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


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclasses.dataclass
class PlanClip:
    label: str
    title: str
    start_time: float
    end_time: float
    duration_seconds: int
    start_prompt: str
    end_prompt: str
    existing_anchor: str


@dataclasses.dataclass
class KeyframeTask:
    task_id: str
    clip_label: str
    clip_title: str
    phase: str
    start_time: float
    end_time: float
    duration_seconds: int
    raw_prompt: str
    existing_anchor: str
    prompt: str
    negative_prompt: str
    seed: int
    attempt: int = 0
    prompt_id: str | None = None
    output_path: str | None = None
    status: str = "planned"
    error: str | None = None
    ai_notes: list[str] = dataclasses.field(default_factory=list)
    reference_requests: list[str] = dataclasses.field(default_factory=list)
    resolved_reference_paths: list[str] = dataclasses.field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class NodeMatch:
    id: str | None = None
    node_type: str | None = None
    title_contains: str | None = None
    type_contains: str | None = None


@dataclasses.dataclass
class WorkflowOverride:
    match: NodeMatch
    widgets: dict[str, Any]


@dataclasses.dataclass
class AIConfig:
    enabled: bool = False
    api_base: str = ""
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    director_session_id: str = "rwbt-main"
    persist_context: bool = True
    set_plan_on_run_start: bool = True
    plan_max_chars: int = 120000
    temperature: float = 0.2
    max_tokens: int = 1200
    interpret_before_generate: bool = True
    analyze_after_generate: bool = True
    max_retries_per_task: int = 2
    require_vision: bool = True


@dataclasses.dataclass
class RunnerConfig:
    prompt_plan_path: str
    output_root: str
    workflow_template_path: str
    comfy_root: str = "."
    comfy_api_url: str = "http://127.0.0.1:18188"
    comfy_api_verify_tls: bool = False
    job_id: str | None = None
    width: int = 1280
    height: int = 720
    steps: int = 32
    cfg: float = 6.0
    sampler_name: str = "dpmpp_2m"
    scheduler: str = "karras"
    denoise: float = 1.0
    model_name: str = "sd_xl_base_1.0.safetensors"
    vae_name: str = ""
    clip_name: str = ""
    seed_strategy: str = "derived"
    base_seed: int = 42
    seed_offset: int = 1009
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    negative_prompt: str = "text, subtitle, logo, watermark, blurry, low quality, deformed"
    include_plan_global_context: bool = True
    include_plan_negative_prompt: bool = True
    stop_on_failure: bool = False
    max_tasks: int | None = None
    max_wait_seconds: float = 900.0
    reference_image_dir: str = ""
    max_reference_images: int = 4
    reference_node_title_keywords: list[str] = dataclasses.field(
        default_factory=lambda: ["reference", "ref", "ipadapter", "adapter image"]
    )
    ai: AIConfig = dataclasses.field(default_factory=AIConfig)
    workflow_overrides: list[WorkflowOverride] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.steps = int(self.steps)
        self.cfg = float(self.cfg)
        self.denoise = float(self.denoise)
        self.max_reference_images = max(0, int(self.max_reference_images))
        self.seed_strategy = str(self.seed_strategy or "derived").strip().lower()
        if self.seed_strategy not in {"random", "deterministic", "derived"}:
            raise ValueError("seed_strategy must be random, deterministic, or derived")
        if isinstance(self.ai, dict):
            self.ai = AIConfig(**self.ai)
        if isinstance(self.workflow_overrides, list):
            converted: list[WorkflowOverride] = []
            for item in self.workflow_overrides:
                if isinstance(item, WorkflowOverride):
                    converted.append(item)
                    continue
                match = item.get("match", {}) if isinstance(item, dict) else {}
                converted.append(
                    WorkflowOverride(
                        match=NodeMatch(
                            id=str(match.get("id")) if match.get("id") is not None else None,
                            node_type=match.get("node_type"),
                            title_contains=match.get("title_contains"),
                            type_contains=match.get("type_contains"),
                        ),
                        widgets=item.get("widgets", {}) if isinstance(item, dict) else {},
                    )
                )
            self.workflow_overrides = converted


class ComfyClient:
    def __init__(self, base_url: str, verify_tls: bool = False):
        self.base_url = base_url.rstrip("/")
        self.verify_tls = verify_tls
        self.session = requests.Session()

    def get_json(self, path: str) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}{path}", timeout=30, verify=self.verify_tls)
        response.raise_for_status()
        return response.json()

    def post_json(self, path: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}{path}", json=payload, timeout=timeout, verify=self.verify_tls
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code} on {path}: {response.text[:4000]}")
        return response.json()

    def convert_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        return self.post_json("/workflow/convert", workflow)

    def queue_prompt(self, prompt: dict[str, Any]) -> str:
        return str(self.post_json("/prompt", {"prompt": prompt})["prompt_id"])

    def wait_for_completion(self, prompt_id: str, max_wait_seconds: float = 900.0, poll_seconds: float = 2.0) -> dict[str, Any]:
        deadline = time.time() + max_wait_seconds
        while time.time() < deadline:
            history = self.get_json(f"/history/{prompt_id}")
            if history and prompt_id in history:
                item = history[prompt_id]
                if item.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"Prompt failed: {item.get('node_errors', {})}")
                return item
            time.sleep(poll_seconds)
        raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


class OpenAICompatClient:
    def __init__(self, config: AIConfig):
        self.config = config
        self.enabled = bool(config.enabled and config.api_base and config.model)
        self.api_key = os.getenv(config.api_key_env, "")
        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Director-Session": str(self.config.director_session_id or "rwbt-main"),
            "X-Director-Persist": "1" if self.config.persist_context else "0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def set_active_plan(self, plan_text: str, plan_id: str) -> bool:
        if not self.enabled:
            return False
        payload = {
            "session_id": str(self.config.director_session_id or "rwbt-main"),
            "plan_id": str(plan_id or "").strip(),
            "plan_text": str(plan_text or ""),
        }
        try:
            response = self.session.post(
                self.config.api_base.rstrip("/") + "/director/plan",
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
        except requests.RequestException:
            return False
        return response.status_code < 400

    def chat_json(self, system_prompt: str, user_prompt: str, image_path: Path | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if not self.api_key:
            raise RuntimeError(
                f"AI enabled but API key env var is missing: {self.config.api_key_env}"
            )

        headers = self._headers()

        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if image_path is not None:
            if not image_path.exists():
                raise FileNotFoundError(f"AI analyze image missing: {image_path}")
            mime = "image/png"
            suffix = image_path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif suffix == ".webp":
                mime = "image/webp"
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                }
            )

        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "session_id": str(self.config.director_session_id or "rwbt-main"),
            "persist_context": bool(self.config.persist_context),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

        response = self.session.post(
            self.config.api_base.rstrip("/") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"AI HTTP {response.status_code}: {response.text[:4000]}")
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return parse_json_object(str(content))


class RWBTPlanParser:
    CLIP_RE = re.compile(
        r"^##\s+Clip\s+(?P<label>.+?)\s+[\u2014-]\s+(?P<start>\d{2}:\d{2})[\u2013-](?P<end>\d{2}:\d{2})\s+[\u2014-]\s+(?P<title>.+?)\s*$",
        re.MULTILINE,
    )

    START_RE = re.compile(
        r"###\s+START\s+keyframe\s+image\s+prompt\s*\n\s*```text\s*\n(?P<prompt>.*?)\n```",
        re.IGNORECASE | re.DOTALL,
    )

    END_RE = re.compile(
        r"###\s+END\s+keyframe\s+image\s+prompt\s*\n\s*```text\s*\n(?P<prompt>.*?)\n```",
        re.IGNORECASE | re.DOTALL,
    )

    ANCHOR_RE = re.compile(r"\*\*Existing anchor image:\*\*\s*(?P<anchor>.+)", re.IGNORECASE)

    def __init__(self, markdown_text: str):
        self.text = markdown_text

    def global_context_text(self) -> str:
        first_clip = self.CLIP_RE.search(self.text)
        head = self.text[: first_clip.start()] if first_clip else self.text
        lines = [line.rstrip() for line in head.splitlines()]
        compact = [line.strip() for line in lines if line.strip()]
        return "\n".join(compact)

    def global_negative_text(self) -> str:
        # RWBT plans usually encode negatives inline; this keeps hook parity with other runners.
        return ""

    def parse_clips(self) -> list[PlanClip]:
        matches = list(self.CLIP_RE.finditer(self.text))
        clips: list[PlanClip] = []
        for idx, match in enumerate(matches):
            block_start = match.end()
            block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(self.text)
            block = self.text[block_start:block_end]

            start_match = self.START_RE.search(block)
            end_match = self.END_RE.search(block)
            if not start_match or not end_match:
                raise ValueError(f"Clip {match.group('label')} is missing START/END keyframe code block prompts")

            anchor_match = self.ANCHOR_RE.search(block)
            anchor = anchor_match.group("anchor").strip() if anchor_match else ""

            start_time = parse_timecode(match.group("start"))
            end_time = parse_timecode(match.group("end"))

            clips.append(
                PlanClip(
                    label=match.group("label").strip(),
                    title=match.group("title").strip(),
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=max(1, int(round(end_time - start_time))),
                    start_prompt=start_match.group("prompt").strip(),
                    end_prompt=end_match.group("prompt").strip(),
                    existing_anchor=anchor,
                )
            )
        return clips


class RWBTKeyframeRunner:
    def __init__(self, config: RunnerConfig):
        self.config = config
        self.comfy_root = self._resolve(config.comfy_root)
        self.prompt_plan_path = self._resolve(config.prompt_plan_path)
        self.workflow_template_path = self._resolve(config.workflow_template_path)
        self.output_root = self._resolve(config.output_root)
        self.reference_image_dir = self._resolve(config.reference_image_dir) if config.reference_image_dir else self.prompt_plan_path.parent

        self.job_id = config.job_id or f"rwbt_keyframes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.job_dir = self.output_root / self.job_id
        self.images_dir = self.job_dir / "images"
        self.manifest_dir = self.job_dir / "manifest"
        self.reference_input_dir = self.comfy_root / "input" / f"{self.job_id}_refs"
        self.state_path = self.job_dir / "job_state.json"
        self.manifest_path = self.manifest_dir / "tasks_manifest.json"

        self.client = ComfyClient(config.comfy_api_url, config.comfy_api_verify_tls)
        self.ai_client = OpenAICompatClient(config.ai)
        self.generated_lookup: dict[str, Path] = {}
        self.anchor_index_lookup: dict[str, Path] = self._index_reference_images(self.reference_image_dir)

    def _resolve(self, value: str) -> Path:
        path = Path(str(value).strip())
        if path.is_absolute():
            return path.resolve()
        return path.resolve() if value == "." else (Path.cwd() / path).resolve()

    def validate(self, check_api: bool = True) -> None:
        if not self.prompt_plan_path.exists():
            raise FileNotFoundError(f"prompt_plan_path does not exist: {self.prompt_plan_path}")
        if not self.workflow_template_path.exists():
            raise FileNotFoundError(f"workflow_template_path does not exist: {self.workflow_template_path}")
        if not self.reference_image_dir.exists():
            raise FileNotFoundError(f"reference_image_dir does not exist: {self.reference_image_dir}")
        if check_api:
            self.client.get_json("/system_stats")

    @staticmethod
    def _index_reference_images(reference_dir: Path) -> dict[str, Path]:
        lookup: dict[str, Path] = {}
        if not reference_dir.exists():
            return lookup
        for path in sorted(reference_dir.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            index_match = re.search(r"\((\d+)\)", path.stem)
            if index_match:
                lookup[index_match.group(1)] = path.resolve()
        return lookup

    @staticmethod
    def _extract_reference_requests(raw_prompt: str) -> list[str]:
        refs: list[str] = []
        # Examples in RWBT prompts: "Previously generated 1A END", "Existing anchor image RWBT (12)"
        for clip_match in re.finditer(r"Previously\s+generated\s+([0-9]+[A-Z])\s+(START|END)", raw_prompt, re.IGNORECASE):
            refs.append(f"generated:{clip_match.group(1).upper()}:{clip_match.group(2).upper()}")
        for anchor_match in re.finditer(r"RWBT\s*\((\d+)\)", raw_prompt, re.IGNORECASE):
            refs.append(f"anchor:{anchor_match.group(1)}")
        # keep stable order while removing duplicates
        deduped: list[str] = []
        for token in refs:
            if token not in deduped:
                deduped.append(token)
        return deduped

    def _copy_ref_to_input(self, source: Path, task: KeyframeTask, slot: int) -> str:
        self.reference_input_dir.mkdir(parents=True, exist_ok=True)
        safe_clip = task.clip_label.replace(" ", "_")
        target = self.reference_input_dir / f"{task.task_id}_{safe_clip}_r{slot:02d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        return f"{self.reference_input_dir.name}/{target.name}"

    def _resolve_references_for_task(self, task: KeyframeTask) -> list[str]:
        resolved: list[str] = []

        # Always include immediate previous accepted frame first when available.
        prev_key = f"{task.clip_label}:START" if task.phase == "END" else ""
        if prev_key and prev_key in self.generated_lookup:
            resolved.append(str(self.generated_lookup[prev_key]))

        for token in task.reference_requests:
            if token.startswith("generated:"):
                _, clip_label, phase = token.split(":", 2)
                key = f"{clip_label}:{phase}"
                ref_path = self.generated_lookup.get(key)
                if ref_path:
                    resolved.append(str(ref_path))
                continue
            if token.startswith("anchor:"):
                _, idx = token.split(":", 1)
                anchor = self.anchor_index_lookup.get(idx)
                if anchor:
                    resolved.append(str(anchor))

        # De-dup and cap at configured maximum.
        unique: list[str] = []
        for item in resolved:
            if item not in unique:
                unique.append(item)
        if self.config.max_reference_images > 0:
            unique = unique[: self.config.max_reference_images]
        task.resolved_reference_paths = unique
        return unique

    def parse_plan(self) -> tuple[str, str, list[PlanClip]]:
        text = self.prompt_plan_path.read_text(encoding="utf-8", errors="replace")
        parser = RWBTPlanParser(text)
        global_context = parser.global_context_text()
        global_negative = parser.global_negative_text()
        clips = parser.parse_clips()
        if self.config.max_tasks is not None:
            max_clips = max(1, int(self.config.max_tasks) // 2)
            clips = clips[:max_clips]
        if not clips:
            raise ValueError("No clips parsed from plan")
        return global_context, global_negative, clips

    def seed_for(self, index: int, prompt: str) -> int:
        if self.config.seed_strategy == "random":
            return random.randint(0, 2_147_483_647)
        if self.config.seed_strategy == "deterministic":
            return self.config.base_seed + index * self.config.seed_offset
        digest = hashlib.sha256(f"{self.config.base_seed}|{index}|{prompt}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def build_tasks(self, global_context: str, global_negative: str, clips: list[PlanClip]) -> list[KeyframeTask]:
        tasks: list[KeyframeTask] = []
        task_index = 0

        for clip in clips:
            for phase, prompt_text in (("START", clip.start_prompt), ("END", clip.end_prompt)):
                task_index += 1
                core_prompt = "\n\n".join(
                    part
                    for part in [
                        global_context if self.config.include_plan_global_context else "",
                        self.config.prompt_prefix.strip(),
                        prompt_text.strip(),
                        self.config.prompt_suffix.strip(),
                    ]
                    if part
                )
                neg_prompt = "\n\n".join(
                    part
                    for part in [
                        self.config.negative_prompt.strip(),
                        global_negative.strip() if self.config.include_plan_negative_prompt else "",
                    ]
                    if part
                )
                task_id = f"{task_index:04d}_{clip.label}_{phase.lower()}"
                tasks.append(
                    KeyframeTask(
                        task_id=task_id,
                        clip_label=clip.label,
                        clip_title=clip.title,
                        phase=phase,
                        start_time=clip.start_time,
                        end_time=clip.end_time,
                        duration_seconds=clip.duration_seconds,
                        raw_prompt=prompt_text.strip(),
                        existing_anchor=clip.existing_anchor,
                        prompt=core_prompt,
                        negative_prompt=neg_prompt,
                        seed=self.seed_for(task_index, core_prompt),
                        reference_requests=self._extract_reference_requests(prompt_text.strip()),
                    )
                )

        if self.config.max_tasks is not None:
            return tasks[: int(self.config.max_tasks)]
        return tasks

    @staticmethod
    def _workflow_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = list(workflow.get("nodes", []))
        for subgraph in (workflow.get("definitions") or {}).get("subgraphs", []):
            nodes.extend(subgraph.get("nodes", []))
        return nodes

    @staticmethod
    def _set_widget(node: dict[str, Any], value: Any, index: int = 0) -> None:
        widgets = node.setdefault("widgets_values", [])
        if not isinstance(widgets, list):
            node["widgets_values"] = []
            widgets = node["widgets_values"]
        while len(widgets) <= index:
            widgets.append(None)
        widgets[index] = value

    def _matches_override(self, node: dict[str, Any], match: NodeMatch) -> bool:
        node_id = str(node.get("id")) if node.get("id") is not None else ""
        node_type = str(node.get("type") or "")
        title = str(node.get("title") or "")

        if match.id and node_id != str(match.id):
            return False
        if match.node_type and node_type != match.node_type:
            return False
        if match.type_contains and match.type_contains.lower() not in node_type.lower():
            return False
        if match.title_contains and match.title_contains.lower() not in title.lower():
            return False
        return True

    def _format_override_value(self, value: Any, context: dict[str, Any]) -> Any:
        if isinstance(value, str):
            try:
                return value.format_map(context)
            except KeyError:
                return value
        return value

    def _apply_custom_overrides(self, nodes: list[dict[str, Any]], context: dict[str, Any]) -> None:
        for override in self.config.workflow_overrides:
            for node in nodes:
                if not self._matches_override(node, override.match):
                    continue
                for index_text, raw_value in override.widgets.items():
                    index = int(index_text)
                    self._set_widget(node, self._format_override_value(raw_value, context), index)

    def _apply_reference_loaders(self, nodes: list[dict[str, Any]], task: KeyframeTask, input_refs: list[str]) -> None:
        if not input_refs:
            return
        ref_nodes: list[dict[str, Any]] = []
        keywords = [k.lower() for k in self.config.reference_node_title_keywords]
        for node in nodes:
            if str(node.get("type") or "") != "LoadImage":
                continue
            title = str(node.get("title") or "").lower()
            if any(keyword in title for keyword in keywords):
                ref_nodes.append(node)

        # Stable ordering by node id gives deterministic slot assignment.
        ref_nodes.sort(key=lambda n: str(n.get("id")))
        for idx, image_name in enumerate(input_refs):
            if idx >= len(ref_nodes):
                break
            self._set_widget(ref_nodes[idx], image_name, 0)

    def patch_workflow(self, task: KeyframeTask, output_prefix: str, input_refs: list[str] | None = None) -> dict[str, Any]:
        workflow = json.loads(self.workflow_template_path.read_text(encoding="utf-8"))
        nodes = self._workflow_nodes(workflow)

        # Generic patching by common Comfy node types/titles so templates stay flexible.
        for node in nodes:
            node_type = str(node.get("type") or "")
            title = str(node.get("title") or "")
            title_lower = title.lower()

            if node_type == "CLIPTextEncode":
                if "negative" in title_lower:
                    self._set_widget(node, task.negative_prompt, 0)
                else:
                    self._set_widget(node, task.prompt, 0)

            if node_type == "PrimitiveStringMultiline":
                if "negative" in title_lower:
                    self._set_widget(node, task.negative_prompt, 0)
                else:
                    self._set_widget(node, task.prompt, 0)

            if node_type == "PrimitiveInt":
                if "width" in title_lower:
                    self._set_widget(node, int(self.config.width), 0)
                elif "height" in title_lower:
                    self._set_widget(node, int(self.config.height), 0)
                elif "steps" in title_lower:
                    self._set_widget(node, int(self.config.steps), 0)

            if node_type == "PrimitiveFloat":
                if "cfg" in title_lower:
                    self._set_widget(node, float(self.config.cfg), 0)
                elif "denoise" in title_lower:
                    self._set_widget(node, float(self.config.denoise), 0)

            if node_type == "EmptyLatentImage":
                self._set_widget(node, int(self.config.width), 0)
                self._set_widget(node, int(self.config.height), 1)

            if node_type == "KSampler":
                # Typical layout: seed, control, steps, cfg, sampler_name, scheduler, denoise
                self._set_widget(node, int(task.seed), 0)
                self._set_widget(node, int(self.config.steps), 2)
                self._set_widget(node, float(self.config.cfg), 3)
                self._set_widget(node, self.config.sampler_name, 4)
                self._set_widget(node, self.config.scheduler, 5)
                self._set_widget(node, float(self.config.denoise), 6)

            # SD Turbo-style templates often use SamplerCustom + SDTurboScheduler + KSamplerSelect.
            if node_type == "SamplerCustom":
                # widgets_values observed in template: [add_noise, seed, control_after_generate, ???]
                self._set_widget(node, int(task.seed), 1)

            if node_type == "SDTurboScheduler":
                # widgets_values observed in template: [steps, denoise]
                self._set_widget(node, int(self.config.steps), 0)
                self._set_widget(node, float(self.config.denoise), 1)

            if node_type == "KSamplerSelect" and self.config.sampler_name:
                self._set_widget(node, self.config.sampler_name, 0)

            if node_type == "RandomNoise":
                self._set_widget(node, int(task.seed), 0)

            if node_type == "CheckpointLoaderSimple" and self.config.model_name:
                self._set_widget(node, self.config.model_name, 0)

            if node_type == "VAELoader" and self.config.vae_name:
                self._set_widget(node, self.config.vae_name, 0)

            if node_type == "CLIPLoader" and self.config.clip_name:
                self._set_widget(node, self.config.clip_name, 0)

            if node_type == "SaveImage":
                self._set_widget(node, output_prefix, 0)

        self._apply_reference_loaders(nodes, task, input_refs or [])

        override_context = {
            "task_id": task.task_id,
            "clip_label": task.clip_label,
            "clip_title": task.clip_title,
            "phase": task.phase,
            "duration_seconds": task.duration_seconds,
            "prompt": task.prompt,
            "negative_prompt": task.negative_prompt,
            "seed": task.seed,
            "width": self.config.width,
            "height": self.config.height,
            "steps": self.config.steps,
            "cfg": self.config.cfg,
            "sampler_name": self.config.sampler_name,
            "scheduler": self.config.scheduler,
            "denoise": self.config.denoise,
            "output_prefix": output_prefix,
            "input_refs": json.dumps(input_refs or [], ensure_ascii=True),
        }
        self._apply_custom_overrides(nodes, override_context)
        return workflow

    def output_image_from_history(self, history: dict[str, Any]) -> Path | None:
        for output in history.get("outputs", {}).values():
            if not isinstance(output, dict):
                continue
            for item in output.get("images", []):
                filename = item.get("filename")
                if not filename:
                    continue
                suffix = Path(str(filename)).suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                subfolder = item.get("subfolder") or ""
                kind = item.get("type") or "output"
                root = self.comfy_root / kind if kind in {"input", "output", "temp"} else self.comfy_root / "output"
                candidate = root / subfolder / filename
                if candidate.exists():
                    return candidate
        return None

    def ai_interpret_task(self, task: KeyframeTask, continuity_memory: list[str]) -> None:
        if not (self.config.ai.enabled and self.config.ai.interpret_before_generate):
            return

        system_prompt = (
            "You are a keyframe prompt engineering assistant for ComfyUI image generation. "
            "Return JSON only. Do not include markdown. "
            "Keep continuity with prior frames and avoid changing identity-defining traits."
        )
        user_prompt = (
            "Interpret and tune this keyframe task. "
            "Output keys: prompt_addendum (string), negative_addendum (string), "
            "steps (int or null), cfg (number or null), denoise (number or null), "
            "seed_jitter (int, default 0), continuity_note (string).\n\n"
            f"Task ID: {task.task_id}\n"
            f"Clip: {task.clip_label} - {task.clip_title}\n"
            f"Phase: {task.phase}\n"
            f"Existing anchor: {task.existing_anchor}\n"
            f"Raw prompt:\n{task.raw_prompt}\n\n"
            f"Current prompt:\n{task.prompt}\n\n"
            f"Current negative prompt:\n{task.negative_prompt}\n\n"
            "Continuity memory (latest accepted):\n"
            + ("\n".join(continuity_memory[-6:]) if continuity_memory else "(none)")
        )

        parsed = self.ai_client.chat_json(system_prompt, user_prompt)
        if not parsed:
            return

        prompt_addendum = str(parsed.get("prompt_addendum", "")).strip()
        negative_addendum = str(parsed.get("negative_addendum", "")).strip()
        continuity_note = str(parsed.get("continuity_note", "")).strip()

        if prompt_addendum:
            task.prompt = f"{task.prompt}\n\n{prompt_addendum}".strip()
            task.ai_notes.append(f"interpret.prompt_addendum: {prompt_addendum}")
        if negative_addendum:
            task.negative_prompt = f"{task.negative_prompt}\n\n{negative_addendum}".strip()
            task.ai_notes.append(f"interpret.negative_addendum: {negative_addendum}")
        if continuity_note:
            task.ai_notes.append(f"interpret.continuity_note: {continuity_note}")

        if parsed.get("steps") is not None:
            self.config.steps = max(1, int(parsed.get("steps")))
            task.ai_notes.append(f"interpret.steps: {self.config.steps}")
        if parsed.get("cfg") is not None:
            self.config.cfg = float(parsed.get("cfg"))
            task.ai_notes.append(f"interpret.cfg: {self.config.cfg}")
        if parsed.get("denoise") is not None:
            self.config.denoise = float(parsed.get("denoise"))
            task.ai_notes.append(f"interpret.denoise: {self.config.denoise}")

        seed_jitter = int(parsed.get("seed_jitter") or 0)
        if seed_jitter:
            task.seed = int((task.seed + seed_jitter) % 2_147_483_647)
            task.ai_notes.append(f"interpret.seed_jitter: {seed_jitter}")

    def ai_analyze_output(
        self,
        task: KeyframeTask,
        image_path: Path,
        continuity_memory: list[str],
    ) -> dict[str, Any]:
        if not (self.config.ai.enabled and self.config.ai.analyze_after_generate):
            return {"pass": True, "summary": "analysis disabled"}

        if self.config.ai.require_vision and not image_path.exists():
            raise FileNotFoundError(f"Cannot analyze missing image: {image_path}")

        system_prompt = (
            "You are a strict continuity and quality checker for sequential keyframe generation. "
            "Return JSON only with keys: pass (bool), summary (string), issues (array of strings), "
            "prompt_correction (string), negative_correction (string), parameter_adjustments (object)."
        )
        user_prompt = (
            "Analyze the generated image against this task and continuity constraints.\n\n"
            f"Task ID: {task.task_id}\n"
            f"Clip: {task.clip_label} - {task.clip_title}\n"
            f"Phase: {task.phase}\n"
            f"Existing anchor: {task.existing_anchor}\n\n"
            f"Expected prompt:\n{task.prompt}\n\n"
            f"Negative prompt:\n{task.negative_prompt}\n\n"
            "Continuity memory (latest accepted):\n"
            + ("\n".join(continuity_memory[-6:]) if continuity_memory else "(none)")
            + "\n\n"
            "If failing, give concise correction deltas only, not a full rewritten prompt."
        )
        parsed = self.ai_client.chat_json(system_prompt, user_prompt, image_path=image_path)
        if not parsed:
            return {"pass": True, "summary": "analysis parse empty"}
        if "pass" not in parsed:
            parsed["pass"] = True
        return parsed

    def render_task(self, task: KeyframeTask, attempt: int) -> Path:
        output_prefix = f"rwbt_keyframes/{self.job_id}/{task.task_id}_a{attempt:02d}"
        resolved_refs = self._resolve_references_for_task(task)
        input_ref_names: list[str] = []
        for slot, ref_path_text in enumerate(resolved_refs, start=1):
            ref_path = Path(ref_path_text)
            if not ref_path.exists():
                continue
            input_ref_names.append(self._copy_ref_to_input(ref_path, task, slot))

        workflow = self.patch_workflow(task, output_prefix, input_ref_names)
        api_prompt = self.client.convert_workflow(workflow)
        prompt_id = self.client.queue_prompt(api_prompt)
        task.prompt_id = prompt_id
        history = self.client.wait_for_completion(prompt_id, max_wait_seconds=self.config.max_wait_seconds)
        generated = self.output_image_from_history(history)
        if generated is None:
            raise RuntimeError("No image output found in Comfy history")
        final_path = self.images_dir / f"{task.task_id}.png"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, final_path)
        return final_path

    def write_manifest(self, tasks: list[KeyframeTask], global_context: str, global_negative: str) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "job_id": self.job_id,
                "updated_at": now_utc(),
                "mode": "rwbt_keyframe_ai",
                "prompt_plan_path": str(self.prompt_plan_path),
                "workflow_template_path": str(self.workflow_template_path),
                "global_context": global_context,
                "global_negative": global_negative,
                "config": dataclasses.asdict(self.config),
                "tasks": [task.as_dict() for task in tasks],
            },
        )

    def _register_active_plan(self, global_context: str, clips: list[PlanClip]) -> None:
        if not (self.config.ai.enabled and self.config.ai.set_plan_on_run_start):
            return

        source_text = self.prompt_plan_path.read_text(encoding="utf-8", errors="replace")
        plan_text = source_text[: max(1, int(self.config.ai.plan_max_chars))]
        clip_summary = "\n".join(
            f"- {clip.label} ({int(clip.start_time)}s-{int(clip.end_time)}s): {clip.title}"
            for clip in clips
        )
        stitched = (
            "RWBT active plan context\n"
            + f"Plan file: {self.prompt_plan_path}\n"
            + "Global context:\n"
            + global_context
            + "\n\nClip sequence:\n"
            + clip_summary
            + "\n\nFull plan body:\n"
            + plan_text
        )
        plan_id = hashlib.sha256(stitched.encode("utf-8")).hexdigest()[:16]
        ok = self.ai_client.set_active_plan(stitched, plan_id=plan_id)
        if ok:
            print(f"[AI] Registered active plan in director session '{self.config.ai.director_session_id}' (plan_id={plan_id})")
        else:
            print("[AI] Director plan registration skipped or unavailable; continuing with direct chat behavior")

    def run(self, dry_run: bool = False) -> dict[str, Any]:
        self.validate(check_api=not dry_run)

        for directory in [self.job_dir, self.images_dir, self.manifest_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        global_context, global_negative, clips = self.parse_plan()
        self._register_active_plan(global_context, clips)
        tasks = self.build_tasks(global_context, global_negative, clips)

        state: dict[str, Any] = {
            "job_id": self.job_id,
            "created_at": now_utc(),
            "updated_at": now_utc(),
            "status": "planned" if dry_run else "running",
            "mode": "rwbt_keyframe_ai",
            "number_of_tasks": len(tasks),
            "completed_tasks": [],
            "failed_tasks": [],
            "dry_run": bool(dry_run),
        }
        atomic_write_json(self.state_path, state)
        self.write_manifest(tasks, global_context, global_negative)

        self.job_dir.joinpath("prompt_plan_source.md").write_text(
            self.prompt_plan_path.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

        if dry_run:
            return {
                "job_id": self.job_id,
                "job_dir": str(self.job_dir),
                "task_count": len(tasks),
                "dry_run": True,
            }

        continuity_memory: list[str] = []

        for task in tasks:
            state["current_task_id"] = task.task_id
            state["updated_at"] = now_utc()
            atomic_write_json(self.state_path, state)

            attempt_success = False
            max_attempts = max(1, int(self.config.ai.max_retries_per_task) + 1)

            for attempt in range(1, max_attempts + 1):
                task.attempt = attempt
                try:
                    self.ai_interpret_task(task, continuity_memory)
                    output_path = self.render_task(task, attempt)
                    analysis = self.ai_analyze_output(task, output_path, continuity_memory)

                    passed = bool(analysis.get("pass", True))
                    summary = str(analysis.get("summary", "")).strip()
                    if summary:
                        task.ai_notes.append(f"analysis.summary: {summary}")

                    if passed:
                        task.status = "completed"
                        task.output_path = str(output_path)
                        self.generated_lookup[f"{task.clip_label}:{task.phase}"] = output_path
                        continuity_memory.append(
                            f"{task.task_id}: {task.clip_label} {task.phase} accepted. {summary or 'pass'}"
                        )
                        state["completed_tasks"].append(task.task_id)
                        attempt_success = True
                        break

                    issues = analysis.get("issues", [])
                    if issues:
                        task.ai_notes.append("analysis.issues: " + " | ".join(str(x) for x in issues))

                    prompt_corr = str(analysis.get("prompt_correction", "")).strip()
                    neg_corr = str(analysis.get("negative_correction", "")).strip()
                    if prompt_corr:
                        task.prompt = f"{task.prompt}\n\nCorrection delta: {prompt_corr}"
                        task.ai_notes.append(f"analysis.prompt_correction: {prompt_corr}")
                    if neg_corr:
                        task.negative_prompt = f"{task.negative_prompt}\n\nCorrection delta: {neg_corr}"
                        task.ai_notes.append(f"analysis.negative_correction: {neg_corr}")

                    adjustments = analysis.get("parameter_adjustments", {})
                    if isinstance(adjustments, dict):
                        if adjustments.get("steps") is not None:
                            self.config.steps = max(1, int(adjustments["steps"]))
                            task.ai_notes.append(f"analysis.steps: {self.config.steps}")
                        if adjustments.get("cfg") is not None:
                            self.config.cfg = float(adjustments["cfg"])
                            task.ai_notes.append(f"analysis.cfg: {self.config.cfg}")
                        if adjustments.get("denoise") is not None:
                            self.config.denoise = float(adjustments["denoise"])
                            task.ai_notes.append(f"analysis.denoise: {self.config.denoise}")

                    task.seed = int((task.seed + self.config.seed_offset + attempt) % 2_147_483_647)

                except Exception as exc:  # noqa: BLE001
                    task.error = str(exc)
                    task.ai_notes.append(f"attempt_error: {task.error}")

                finally:
                    self.write_manifest(tasks, global_context, global_negative)
                    atomic_write_json(self.state_path, state)

            if not attempt_success:
                task.status = "failed"
                state["failed_tasks"].append(task.task_id)
                if self.config.stop_on_failure:
                    state["status"] = "failed"
                    state["updated_at"] = now_utc()
                    atomic_write_json(self.state_path, state)
                    self.write_manifest(tasks, global_context, global_negative)
                    raise RuntimeError(f"Task failed and stop_on_failure is enabled: {task.task_id}")

            self.write_manifest(tasks, global_context, global_negative)
            state["updated_at"] = now_utc()
            atomic_write_json(self.state_path, state)

        state["status"] = "completed" if not state["failed_tasks"] else "failed"
        state["updated_at"] = now_utc()
        atomic_write_json(self.state_path, state)
        self.write_manifest(tasks, global_context, global_negative)

        return {
            "job_id": self.job_id,
            "job_dir": str(self.job_dir),
            "task_count": len(tasks),
            "completed": len(state["completed_tasks"]),
            "failed": len(state["failed_tasks"]),
            "status": state["status"],
        }


def load_config(path: Path) -> RunnerConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RunnerConfig(**raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RWBT keyframe plan with optional AI interpretation/correction")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = RWBTKeyframeRunner(load_config(args.config)).run(dry_run=args.dry_run)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
