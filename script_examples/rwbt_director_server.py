#!/usr/bin/env python3
"""Persistent local director service for RWBT jobs.

This service exposes an OpenAI-compatible endpoint at /v1/chat/completions
and proxies requests to an upstream OpenAI-compatible model server while
persisting per-session chat memory on disk.

Design goals:
- Stay alive independently of ComfyUI page reloads.
- Keep director memory across separate runner executions.
- Avoid non-stdlib dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib import error, request


def now_ts() -> int:
    return int(time.time())


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        content_length = 0
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclass
class DirectorState:
    state_path: Path
    max_messages_per_session: int = 24
    lock: threading.Lock = field(default_factory=threading.Lock)
    sessions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)
    started_at: int = field(default_factory=now_ts)

    def load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("sessions"), dict):
                sessions: dict[str, list[dict[str, Any]]] = {}
                for key, value in payload["sessions"].items():
                    if isinstance(key, str) and isinstance(value, list):
                        sessions[key] = [item for item in value if isinstance(item, dict)]
                self.sessions = sessions
            if isinstance(payload, dict) and isinstance(payload.get("plans"), dict):
                plans: dict[str, dict[str, Any]] = {}
                for key, value in payload["plans"].items():
                    if isinstance(key, str) and isinstance(value, dict):
                        plan_text = str(value.get("plan_text") or "").strip()
                        if plan_text:
                            plans[key] = {
                                "plan_text": plan_text,
                                "plan_id": str(value.get("plan_id") or "").strip(),
                                "updated_at": int(value.get("updated_at") or now_ts()),
                            }
                self.plans = plans
        except (OSError, json.JSONDecodeError):
            # Keep service available even if persisted state is malformed.
            self.sessions = {}
            self.plans = {}

    def save(self) -> None:
        atomic_write_json(
            self.state_path,
            {
                "saved_at": now_ts(),
                "started_at": self.started_at,
                "sessions": self.sessions,
                "plans": self.plans,
            },
        )

    def get_context(self, session_id: str) -> list[dict[str, Any]]:
        with self.lock:
            return deepcopy(self.sessions.get(session_id, []))

    def update_context(self, session_id: str, new_messages: list[dict[str, Any]]) -> None:
        with self.lock:
            existing = self.sessions.get(session_id, [])
            merged = existing + [m for m in new_messages if isinstance(m, dict)]
            if len(merged) > self.max_messages_per_session:
                merged = merged[-self.max_messages_per_session :]
            self.sessions[session_id] = merged
            self.save()

    def reset_session(self, session_id: str) -> None:
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.plans:
                del self.plans[session_id]
            self.save()

    def set_plan(self, session_id: str, plan_text: str, plan_id: str = "") -> None:
        plan_text = str(plan_text or "").strip()
        with self.lock:
            if not plan_text:
                if session_id in self.plans:
                    del self.plans[session_id]
                    self.save()
                return
            self.plans[session_id] = {
                "plan_text": plan_text,
                "plan_id": str(plan_id or "").strip(),
                "updated_at": now_ts(),
            }
            self.save()

    def get_plan(self, session_id: str) -> dict[str, Any] | None:
        with self.lock:
            plan = self.plans.get(session_id)
            return deepcopy(plan) if isinstance(plan, dict) else None

    def clear_plan(self, session_id: str) -> None:
        with self.lock:
            if session_id in self.plans:
                del self.plans[session_id]
                self.save()


@dataclass
class DirectorConfig:
    host: str
    port: int
    upstream_api_base: str
    default_model: str
    api_key_env: str
    default_session_id: str
    state_path: Path


class DirectorProxy:
    def __init__(self, cfg: DirectorConfig, state: DirectorState):
        self.cfg = cfg
        self.state = state

    def _upstream_url(self) -> str:
        return self.cfg.upstream_api_base.rstrip("/") + "/chat/completions"

    def _upstream_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.cfg.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _extract_assistant_message(response_json: dict[str, Any]) -> dict[str, Any] | None:
        try:
            msg = response_json["choices"][0]["message"]
            if isinstance(msg, dict):
                return msg
        except (KeyError, IndexError, TypeError):
            return None
        return None

    def chat(self, body: dict[str, Any], session_id: str, persist_context: bool) -> tuple[int, dict[str, Any]]:
        incoming_messages = body.get("messages")
        if not isinstance(incoming_messages, list) or not incoming_messages:
            return 400, {"error": {"message": "messages must be a non-empty array"}}

        proxied_body = deepcopy(body)
        proxied_body["model"] = str(body.get("model") or self.cfg.default_model)

        plan_system_message: list[dict[str, Any]] = []
        plan = self.state.get_plan(session_id)
        if plan and str(plan.get("plan_text") or "").strip():
            plan_id = str(plan.get("plan_id") or "").strip()
            plan_title = f"Active plan id: {plan_id}\n" if plan_id else ""
            plan_system_message = [
                {
                    "role": "system",
                    "content": (
                        "You must follow the active production plan unless user explicitly replaces it.\n"
                        + plan_title
                        + "Active production plan:\n"
                        + str(plan["plan_text"])
                    ),
                }
            ]

        if persist_context:
            prior = self.state.get_context(session_id)
            proxied_body["messages"] = plan_system_message + prior + incoming_messages
        else:
            proxied_body["messages"] = plan_system_message + incoming_messages

        req = request.Request(
            self._upstream_url(),
            data=json.dumps(proxied_body, ensure_ascii=True).encode("utf-8"),
            headers=self._upstream_headers(),
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=240) as resp:
                status = int(resp.status)
                raw = resp.read()
        except error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            return int(exc.code), {"error": {"message": f"upstream HTTP {exc.code}: {payload[:2000]}"}}
        except Exception as exc:  # pragma: no cover - networking dependent
            return 502, {"error": {"message": f"upstream request failed: {exc}"}}

        try:
            upstream_json = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return 502, {"error": {"message": "upstream returned non-JSON response"}}

        if persist_context and status < 400:
            assistant = self._extract_assistant_message(upstream_json)
            additions: list[dict[str, Any]] = []
            for msg in incoming_messages:
                if isinstance(msg, dict) and msg.get("role") in {"system", "user"}:
                    additions.append(msg)
            if assistant is not None:
                additions.append(assistant)
            if additions:
                self.state.update_context(session_id, additions)

        return status, upstream_json


def build_handler(proxy: DirectorProxy, cfg: DirectorConfig, state: DirectorState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")

            if path == "/health":
                json_response(
                    self,
                    200,
                    {
                        "service": "rwbt-director",
                        "status": "ok",
                        "started_at": state.started_at,
                        "uptime_seconds": now_ts() - state.started_at,
                        "sessions": len(state.sessions),
                        "upstream_api_base": cfg.upstream_api_base,
                        "default_model": cfg.default_model,
                    },
                )
                return

            if path == "/director/plan":
                qs = parse_qs(parsed.query or "")
                session_id = str((qs.get("session_id") or [cfg.default_session_id])[0]).strip() or cfg.default_session_id
                plan = state.get_plan(session_id)
                json_response(
                    self,
                    200,
                    {
                        "session_id": session_id,
                        "has_plan": bool(plan),
                        "plan": plan or {},
                    },
                )
                return

            json_response(self, 404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            body = parse_json_body(self)

            if path == "/director/reset":
                session_id = str(body.get("session_id") or cfg.default_session_id)
                state.reset_session(session_id)
                json_response(self, 200, {"ok": True, "session_id": session_id})
                return

            if path == "/director/plan":
                session_id = str(body.get("session_id") or cfg.default_session_id).strip() or cfg.default_session_id
                plan_text = str(body.get("plan_text") or "")
                plan_id = str(body.get("plan_id") or "")
                state.set_plan(session_id=session_id, plan_text=plan_text, plan_id=plan_id)
                json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "session_id": session_id,
                        "has_plan": bool(plan_text.strip()),
                        "plan_id": plan_id,
                    },
                )
                return

            if path == "/director/clear_plan":
                session_id = str(body.get("session_id") or cfg.default_session_id).strip() or cfg.default_session_id
                state.clear_plan(session_id)
                json_response(self, 200, {"ok": True, "session_id": session_id})
                return

            if path != "/v1/chat/completions":
                json_response(self, 404, {"error": {"message": "not found"}})
                return

            header_session_id = self.headers.get("X-Director-Session", "").strip()
            body_session_id = str(body.get("session_id") or "").strip()
            session_id = header_session_id or body_session_id or cfg.default_session_id

            persist_header = self.headers.get("X-Director-Persist", "1").strip().lower()
            persist_context = persist_header not in {"0", "false", "no"}
            if isinstance(body.get("persist_context"), bool):
                persist_context = bool(body.get("persist_context"))

            status, payload = proxy.chat(body, session_id=session_id, persist_context=persist_context)
            json_response(self, status, payload)

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # Keep output concise for long-running service logs.
            sys_line = "%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args)
            try:
                print(sys_line, end="")
            except Exception:
                pass

    return Handler


def parse_args() -> DirectorConfig:
    parser = argparse.ArgumentParser(description="Persistent RWBT director proxy service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--upstream-api-base", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--default-session-id", default="rwbt-main")
    parser.add_argument("--state-path", default="/workspace/ComfyUI/output/rwbt_director_state/state.json")
    args = parser.parse_args()

    return DirectorConfig(
        host=str(args.host),
        port=int(args.port),
        upstream_api_base=str(args.upstream_api_base),
        default_model=str(args.model),
        api_key_env=str(args.api_key_env),
        default_session_id=str(args.default_session_id),
        state_path=Path(str(args.state_path)).resolve(),
    )


def main() -> int:
    cfg = parse_args()
    state = DirectorState(state_path=cfg.state_path)
    state.load()
    proxy = DirectorProxy(cfg=cfg, state=state)
    handler_cls = build_handler(proxy=proxy, cfg=cfg, state=state)

    httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler_cls)
    print(
        "[rwbt-director] listening on http://%s:%d | upstream=%s | state=%s"
        % (cfg.host, cfg.port, cfg.upstream_api_base, str(cfg.state_path))
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[rwbt-director] stopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
