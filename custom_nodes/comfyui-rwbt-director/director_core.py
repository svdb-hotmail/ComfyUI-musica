from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

import folder_paths


DEFAULT_DIRECTOR_API_BASE = "http://127.0.0.1:8099"
AUTOSTART_ENV = "RWBT_DIRECTOR_AUTOSTART"
_last_start_attempt_ts = 0.0


def director_api_base() -> str:
    return str(os.getenv("RWBT_DIRECTOR_API_BASE", DEFAULT_DIRECTOR_API_BASE)).rstrip("/")


def _json_request(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
    ensure_server: bool = True,
) -> dict[str, Any]:
    if ensure_server:
        ensure_director_running()

    body: bytes | None = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    req = request.Request(url, data=body, method=method.upper(), headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}: {text[:2000]}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def director_health() -> dict[str, Any]:
    return _json_request("GET", f"{director_api_base()}/health", timeout=15.0, ensure_server=False)


def director_get_plan(session_id: str) -> dict[str, Any]:
    sid = session_id.strip() or "rwbt-main"
    return _json_request("GET", f"{director_api_base()}/director/plan?session_id={sid}", timeout=20.0)


def director_set_plan(session_id: str, plan_text: str, plan_id: str = "") -> dict[str, Any]:
    payload = {
        "session_id": session_id.strip() or "rwbt-main",
        "plan_id": plan_id.strip(),
        "plan_text": plan_text,
    }
    return _json_request("POST", f"{director_api_base()}/director/plan", payload=payload, timeout=20.0)


def director_clear_plan(session_id: str) -> dict[str, Any]:
    payload = {"session_id": session_id.strip() or "rwbt-main"}
    return _json_request("POST", f"{director_api_base()}/director/clear_plan", payload=payload, timeout=20.0)


def director_chat(
    session_id: str,
    user_message: str,
    system_prompt: str = "",
    persist_context: bool = True,
    model: str = "",
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "session_id": session_id.strip() or "rwbt-main",
        "persist_context": bool(persist_context),
        "model": model.strip(),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": messages,
    }
    return _json_request("POST", f"{director_api_base()}/v1/chat/completions", payload=payload, timeout=240.0)


def rwbt_jobs_root(output_root: str = "") -> Path:
    if output_root.strip():
        return Path(output_root.strip()).expanduser().resolve()
    return (Path(folder_paths.get_output_directory()) / "rwbt_jobs").resolve()


def list_rwbt_jobs(output_root: str = "") -> list[dict[str, Any]]:
    root = rwbt_jobs_root(output_root)
    if not root.exists():
        return []
    jobs: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        state_path = child / "job_state.json"
        manifest_path = child / "manifest" / "tasks_manifest.json"
        status = "unknown"
        completed = 0
        failed = 0
        updated_at = ""
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                status = str(data.get("status") or status)
                completed = len(data.get("completed_tasks") or [])
                failed = len(data.get("failed_tasks") or [])
                updated_at = str(data.get("updated_at") or "")
            except Exception:
                pass
        jobs.append(
            {
                "job_id": child.name,
                "job_dir": str(child),
                "status": status,
                "completed": completed,
                "failed": failed,
                "updated_at": updated_at,
                "has_state": state_path.exists(),
                "has_manifest": manifest_path.exists(),
            }
        )
    return jobs


def list_job_outputs(job_id: str, output_root: str = "", limit: int = 40) -> dict[str, Any]:
    root = rwbt_jobs_root(output_root)
    job_dir = root / job_id
    images_dir = job_dir / "images"
    manifest_path = job_dir / "manifest" / "tasks_manifest.json"
    state_path = job_dir / "job_state.json"

    images: list[str] = []
    if images_dir.exists():
        files = sorted(images_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for file in files[: max(1, int(limit))]:
            if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                images.append(str(file))

    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "state_path": str(state_path),
        "manifest_path": str(manifest_path),
        "images": images,
    }


def extract_assistant_text(chat_response: dict[str, Any]) -> str:
    try:
        msg = chat_response["choices"][0]["message"]["content"]
        if isinstance(msg, str):
            return msg
        if isinstance(msg, list):
            parts: list[str] = []
            for item in msg:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
    except Exception:
        pass
    return ""


def resolve_preview_image_path(image_path: str, output_root: str = "") -> Path | None:
    raw = image_path.strip()
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (rwbt_jobs_root(output_root) / candidate).resolve()
    else:
        candidate = candidate.resolve()

    root = rwbt_jobs_root(output_root)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None

    if not candidate.exists() or not candidate.is_file():
        return None

    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None
    return candidate


def preview_image_mime(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    return "application/octet-stream"


def _director_probe(timeout: float = 1.2) -> bool:
    try:
        req = request.Request(f"{director_api_base()}/health", method="GET")
        with request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


def comfy_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def director_server_script_path() -> Path:
    return comfy_root_dir() / "script_examples" / "rwbt_director_server.py"


def start_director_server_background() -> dict[str, Any]:
    global _last_start_attempt_ts
    if _director_probe(timeout=1.0):
        return {"started": False, "online": True, "reason": "already_running"}

    now = time.time()
    if (now - _last_start_attempt_ts) < 2.0:
        return {"started": False, "online": False, "reason": "throttled"}
    _last_start_attempt_ts = now

    script = director_server_script_path()
    if not script.exists():
        return {"started": False, "online": False, "error": f"Missing script: {script}"}

    logs_dir = Path(folder_paths.get_output_directory())
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "rwbt_director_server.log"

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [sys.executable, str(script)]

    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            if os.name == "nt":
                subprocess.Popen(
                    cmd,
                    cwd=str(comfy_root_dir()),
                    env=env,
                    stdout=log_file,
                    stderr=log_file,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                )
            else:
                subprocess.Popen(
                    cmd,
                    cwd=str(comfy_root_dir()),
                    env=env,
                    stdout=log_file,
                    stderr=log_file,
                    start_new_session=True,
                )
    except Exception as exc:  # noqa: BLE001
        return {"started": False, "online": False, "error": str(exc)}

    for _ in range(16):
        if _director_probe(timeout=1.0):
            return {"started": True, "online": True, "log_path": str(log_path)}
        time.sleep(0.5)
    return {"started": True, "online": False, "log_path": str(log_path)}


def ensure_director_running() -> dict[str, Any]:
    autostart = str(os.getenv(AUTOSTART_ENV, "1")).strip().lower() not in {"0", "false", "no"}
    if _director_probe(timeout=1.0):
        return {"online": True, "started": False}
    if not autostart:
        return {"online": False, "started": False, "error": "director_offline_and_autostart_disabled"}
    return start_director_server_background()
