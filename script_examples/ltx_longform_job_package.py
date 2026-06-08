from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from script_examples.longform_ltx23_runner import LongformLTX23Runner, LTXJobConfig


DEFAULT_TEMPLATE = "script_examples/workflows/video_ltx2_3_ia2v.json"
SAFE_JOB_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _as_object(value: Any, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    raise ValueError(f"{field_name} must be an object")


def _as_list(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    raise ValueError(f"{field_name} must be a list")


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "none", "null", ""}:
        return False
    raise ValueError(f"{field_name} must be a boolean or common boolean string/number")


def _config_value(settings: dict[str, Any], package: dict[str, Any], field_name: str, default: Any) -> Any:
    if field_name in settings:
        return settings[field_name]
    return package.get(field_name, default)


def _safe_job_id(value: str, fallback: str) -> str:
    candidate = _first_text(value, fallback)
    candidate = candidate.replace("\\", "/").split("/")[-1]
    candidate = SAFE_JOB_ID_PATTERN.sub("_", candidate).strip("._-")
    return candidate or fallback


def _seconds_to_timestamp(seconds: Any) -> str:
    total = max(0.0, float(seconds))
    whole = int(total)
    millis = int(round((total - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    minutes, sec = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    base = f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes}:{sec:02d}"
    if millis:
        return f"{base}.{millis:03d}".rstrip("0")
    return base


def _plan_from_shots(shots: list[Any]) -> str:
    blocks: list[str] = []
    for index, raw_shot in enumerate(shots, start=1):
        shot = _as_object(raw_shot, f"shots[{index - 1}]")
        start = shot.get("start", shot.get("start_time"))
        end = shot.get("end", shot.get("end_time"))
        if start is None or end is None:
            raise ValueError(f"shots[{index - 1}] must include start/end or start_time/end_time")
        label = _first_text(shot.get("id"), shot.get("shot_id"), shot.get("label"), f"Clip {index}")
        heading_label = label if re.search(r"\bclip\b", label, re.IGNORECASE) else f"Clip {index} {label}"
        prompt = _first_text(shot.get("prompt"), shot.get("summary"), shot.get("description"))
        if not prompt:
            raise ValueError(f"shots[{index - 1}] must include prompt, summary, or description")
        duration = max(0.0, float(end) - float(start))
        blocks.append(
            f"### {heading_label} - {_seconds_to_timestamp(start)}-{_seconds_to_timestamp(end)} - {duration:g}s\n"
            f"```text\n{prompt}\n```"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _image_paths(package: dict[str, Any]) -> list[str]:
    paths = [_first_text(path) for path in _as_list(package.get("image_paths"), "image_paths")]
    images = _as_list(package.get("images"), "images")
    for index, raw_image in enumerate(images):
        if isinstance(raw_image, str):
            path = raw_image.strip()
        else:
            image = _as_object(raw_image, f"images[{index}]")
            path = _first_text(image.get("path"), image.get("file_path"), image.get("local_path"), image.get("uri"))
        if path:
            paths.append(path)
    keyframes = _as_list(package.get("keyframes"), "keyframes")
    for index, raw_keyframe in enumerate(keyframes):
        keyframe = _as_object(raw_keyframe, f"keyframes[{index}]")
        path = _first_text(keyframe.get("path"), keyframe.get("image_path"), keyframe.get("file_path"), keyframe.get("local_path"), keyframe.get("uri"))
        if path:
            paths.append(path)
    unique_paths: list[str] = []
    for path in paths:
        if path and path not in unique_paths:
            unique_paths.append(path)
    if not unique_paths:
        raise ValueError("package must include at least one image path in image_paths, images, or keyframes")
    return unique_paths


def package_to_runner_config(package: dict[str, Any], *, comfy_root: str | Path = ".") -> tuple[dict[str, Any], str]:
    settings = _as_object(package.get("settings") or package.get("render_settings"), "settings")
    audio = _as_object(package.get("audio"), "audio")
    job_id = _safe_job_id(_first_text(package.get("job_id"), package.get("id")), "ltx_job")
    audio_path = _first_text(package.get("audio_path"), audio.get("path"), audio.get("file_path"), audio.get("local_path"), audio.get("uri"))
    if not audio_path:
        raise ValueError("package must include audio_path or audio.path")

    prompt_plan = _first_text(package.get("prompt_plan_text"), package.get("prompt_plan"), package.get("plan_markdown"))
    if not prompt_plan:
        prompt_plan = _plan_from_shots(_as_list(package.get("shots"), "shots"))
    if not prompt_plan.strip():
        raise ValueError("package must include prompt_plan text or shots")

    output_root = _first_text(settings.get("output_root"), package.get("output_root"), "output/longform_ltx23")
    config = {
        "audio_path": audio_path,
        "image_paths": _image_paths(package),
        "prompt_plan_path": "",
        "output_root": output_root,
        "comfy_api_url": _first_text(settings.get("comfy_api_url"), package.get("comfy_api_url"), "http://127.0.0.1:18188"),
        "comfy_api_verify_tls": _coerce_bool(_config_value(settings, package, "comfy_api_verify_tls", False), "comfy_api_verify_tls"),
        "comfy_root": str(comfy_root),
        "workflow_template_path": _first_text(settings.get("workflow_template_path"), package.get("workflow_template_path"), DEFAULT_TEMPLATE),
        "renderer": _first_text(settings.get("renderer"), package.get("renderer"), "ia2v"),
        "global_style_prompt": _first_text(settings.get("global_style_prompt"), package.get("global_style_prompt"), LTXJobConfig.global_style_prompt),
        "negative_prompt": _first_text(settings.get("negative_prompt"), package.get("negative_prompt"), LTXJobConfig.negative_prompt),
        "shot_duration_seconds": float(settings.get("shot_duration_seconds", package.get("shot_duration_seconds", 6.0))),
        "max_shots": settings.get("max_shots", package.get("max_shots")),
        "width": int(settings.get("width", package.get("width", 1280))),
        "height": int(settings.get("height", package.get("height", 720))),
        "fps": int(settings.get("fps", package.get("fps", 24))),
        "seed_strategy": _first_text(settings.get("seed_strategy"), package.get("seed_strategy"), "derived"),
        "base_seed": int(settings.get("base_seed", package.get("base_seed", 42))),
        "seed_offset": int(settings.get("seed_offset", package.get("seed_offset", 1009))),
        "use_previous_final_frame": _coerce_bool(_config_value(settings, package, "use_previous_final_frame", True), "use_previous_final_frame"),
        "prompt_enhance": _coerce_bool(_config_value(settings, package, "prompt_enhance", True), "prompt_enhance"),
        "enable_upscale": _coerce_bool(_config_value(settings, package, "enable_upscale", False), "enable_upscale"),
        "enable_voice_reference": _coerce_bool(_config_value(settings, package, "enable_voice_reference", False), "enable_voice_reference"),
        "resume": _coerce_bool(_config_value(settings, package, "resume", True), "resume"),
        "overwrite": _coerce_bool(_config_value(settings, package, "overwrite", False), "overwrite"),
        "stop_on_failure": _coerce_bool(_config_value(settings, package, "stop_on_failure", False), "stop_on_failure"),
        "final_concat": _coerce_bool(_config_value(settings, package, "final_concat", True), "final_concat"),
        "ffmpeg_crf": int(settings.get("ffmpeg_crf", package.get("ffmpeg_crf", 18))),
        "job_id": job_id or None,
        "resume_job_dir": settings.get("resume_job_dir", package.get("resume_job_dir")),
    }
    if config["max_shots"] is not None:
        config["max_shots"] = int(config["max_shots"])
        if config["max_shots"] <= 0:
            config["max_shots"] = None
    return config, prompt_plan


def materialize_package(package_path: Path, output_config_path: Path | None = None, *, comfy_root: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if not isinstance(package, dict):
        raise ValueError("package JSON must be an object")
    root = Path(comfy_root) if comfy_root is not None else Path.cwd()
    config, prompt_plan = package_to_runner_config(package, comfy_root=root)
    output_root = Path(str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = (root / output_root).resolve()
    job_id = _safe_job_id(str(config.get("job_id") or ""), package_path.stem)
    config["job_id"] = job_id
    config_dir = output_root / "_package_configs"
    prompt_plan_path = config_dir / f"{job_id}_prompt_plan.md"
    prompt_plan_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_plan_path.write_text(prompt_plan, encoding="utf-8")
    config["prompt_plan_path"] = str(prompt_plan_path)
    if output_config_path is None:
        output_config_path = config_dir / f"{job_id}.json"
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_config_path, config


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a StoryDirector LTX longform job package into a runner config")
    parser.add_argument("--package", required=True, type=Path, help="Path to LtxLongformJob JSON package")
    parser.add_argument("--output-config", type=Path, help="Where to write the generated longform_ltx23_runner config")
    parser.add_argument("--comfy-root", type=Path, default=Path.cwd(), help="ComfyUI root used for resolving relative runner paths")
    parser.add_argument("--run", action="store_true", help="Run longform_ltx23_runner after writing the config")
    parser.add_argument("--dry-run", action="store_true", help="Pass dry-run through to the LTX runner when --run is set")
    args = parser.parse_args()

    config_path, config = materialize_package(args.package, args.output_config, comfy_root=args.comfy_root)
    result: dict[str, Any] = {"config_path": str(config_path), "job_id": config.get("job_id")}
    if args.run:
        result["runner"] = LongformLTX23Runner(LTXJobConfig(**config)).run(dry_run=args.dry_run)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
