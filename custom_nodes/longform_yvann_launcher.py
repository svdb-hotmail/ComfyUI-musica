from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from aiohttp import web

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


def _write_config(config_path: Path, config: dict[str, object]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


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
                "chunk_duration_seconds": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "overlap_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3599.0, "step": 1.0}),
                "images_per_chunk": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "image_backend": ("COMBO", {"options": ["procedural", "comfy_api"]}),
                "continuity_mode": ("COMBO", {"options": ["independent", "style", "carry"]}),
                "seed_strategy": ("COMBO", {"options": ["deterministic", "derived", "random"]}),
                "base_seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1}),
                "resume": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "stop_on_failure": ("BOOLEAN", {"default": False}),
                "final_concat": ("BOOLEAN", {"default": True}),
                "yvann_render_fps": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "yvann_min_frames": ("INT", {"default": 24, "min": 8, "max": 1024, "step": 1}),
                "yvann_max_frames": ("INT", {"default": 192, "min": 8, "max": 2048, "step": 1}),
                "max_chunks": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
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
        images_per_chunk,
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
            "image_backend": image_backend,
            "images_per_chunk": int(images_per_chunk),
            "image_width": 640,
            "image_height": 360,
            "comfy_t2i_checkpoint": "DreamShaper_8_pruned.safetensors",
            "comfy_t2i_steps": 6,
            "comfy_t2i_cfg": 4.0,
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongformYvannLauncher": "Yvann Longform Launcher",
}
