from __future__ import annotations

import json
from aiohttp import web

from server import PromptServer

from .director_core import (
    director_chat,
    director_clear_plan,
    director_get_plan,
    director_health,
    director_set_plan,
    ensure_director_running,
    extract_assistant_text,
    list_job_outputs,
    list_rwbt_jobs,
    preview_image_mime,
    resolve_preview_image_path,
)


class RWBTDirectorSetPlan:
    CATEGORY = "RWBT/Director"
    FUNCTION = "set_plan"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("status", "ok")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "rwbt-main"}),
                "plan_id": ("STRING", {"default": ""}),
                "plan_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def set_plan(self, session_id: str, plan_id: str, plan_text: str):
        response = director_set_plan(session_id=session_id, plan_text=plan_text, plan_id=plan_id)
        ok = bool(response.get("ok", False)) and "error" not in response
        status = response.get("error") if "error" in response else json.dumps(response, ensure_ascii=True)
        return (str(status), ok)


class RWBTDirectorGetPlan:
    CATEGORY = "RWBT/Director"
    FUNCTION = "get_plan"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("plan_text", "plan_id", "has_plan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "rwbt-main"}),
            }
        }

    def get_plan(self, session_id: str):
        response = director_get_plan(session_id=session_id)
        if "error" in response:
            return ("", "", False)
        plan = response.get("plan") if isinstance(response.get("plan"), dict) else {}
        plan_text = str(plan.get("plan_text") or "")
        plan_id = str(plan.get("plan_id") or "")
        return (plan_text, plan_id, bool(plan_text.strip()))


class RWBTDirectorClearPlan:
    CATEGORY = "RWBT/Director"
    FUNCTION = "clear_plan"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "rwbt-main"}),
            }
        }

    def clear_plan(self, session_id: str):
        response = director_clear_plan(session_id=session_id)
        return (json.dumps(response, ensure_ascii=True),)


class RWBTDirectorChat:
    CATEGORY = "RWBT/Director"
    FUNCTION = "chat"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("assistant_text", "raw_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "rwbt-main"}),
                "user_message": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "persist_context": ("BOOLEAN", {"default": True}),
                "model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1200, "min": 1, "max": 16384}),
            }
        }

    def chat(
        self,
        session_id: str,
        user_message: str,
        system_prompt: str,
        persist_context: bool,
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        response = director_chat(
            session_id=session_id,
            user_message=user_message,
            system_prompt=system_prompt,
            persist_context=bool(persist_context),
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        assistant_text = extract_assistant_text(response)
        return (assistant_text, json.dumps(response, ensure_ascii=True))


class RWBTDirectorListJobs:
    CATEGORY = "RWBT/Director"
    FUNCTION = "list_jobs"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("jobs_json",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_root": ("STRING", {"default": ""}),
            }
        }

    def list_jobs(self, output_root: str):
        jobs = list_rwbt_jobs(output_root=output_root)
        return (json.dumps(jobs, ensure_ascii=True),)


class RWBTDirectorListJobOutputs:
    CATEGORY = "RWBT/Director"
    FUNCTION = "list_outputs"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("outputs_json",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_id": ("STRING", {"default": ""}),
                "output_root": ("STRING", {"default": ""}),
                "limit": ("INT", {"default": 40, "min": 1, "max": 500}),
            }
        }

    def list_outputs(self, job_id: str, output_root: str, limit: int):
        data = list_job_outputs(job_id=job_id.strip(), output_root=output_root, limit=int(limit))
        return (json.dumps(data, ensure_ascii=True),)


NODE_CLASS_MAPPINGS = {
    "RWBTDirectorSetPlan": RWBTDirectorSetPlan,
    "RWBTDirectorGetPlan": RWBTDirectorGetPlan,
    "RWBTDirectorClearPlan": RWBTDirectorClearPlan,
    "RWBTDirectorChat": RWBTDirectorChat,
    "RWBTDirectorListJobs": RWBTDirectorListJobs,
    "RWBTDirectorListJobOutputs": RWBTDirectorListJobOutputs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RWBTDirectorSetPlan": "RWBT Director Set Plan",
    "RWBTDirectorGetPlan": "RWBT Director Get Plan",
    "RWBTDirectorClearPlan": "RWBT Director Clear Plan",
    "RWBTDirectorChat": "RWBT Director Chat",
    "RWBTDirectorListJobs": "RWBT Director List Jobs",
    "RWBTDirectorListJobOutputs": "RWBT Director List Job Outputs",
}

WEB_DIRECTORY = "./web/js"


@PromptServer.instance.routes.get("/rwbt_director_ui/health")
async def rwbt_director_ui_health(request):
    ensure_director_running()
    return web.json_response(director_health())


@PromptServer.instance.routes.get("/rwbt_director_ui/plan")
async def rwbt_director_ui_get_plan(request):
    ensure_director_running()
    session_id = str(request.rel_url.query.get("session_id") or "rwbt-main")
    return web.json_response(director_get_plan(session_id=session_id))


@PromptServer.instance.routes.post("/rwbt_director_ui/plan")
async def rwbt_director_ui_set_plan(request):
    ensure_director_running()
    payload = await request.json()
    session_id = str(payload.get("session_id") or "rwbt-main")
    plan_id = str(payload.get("plan_id") or "")
    plan_text = str(payload.get("plan_text") or "")
    return web.json_response(director_set_plan(session_id=session_id, plan_text=plan_text, plan_id=plan_id))


@PromptServer.instance.routes.post("/rwbt_director_ui/clear_plan")
async def rwbt_director_ui_clear_plan(request):
    ensure_director_running()
    payload = await request.json()
    session_id = str(payload.get("session_id") or "rwbt-main")
    return web.json_response(director_clear_plan(session_id=session_id))


@PromptServer.instance.routes.post("/rwbt_director_ui/chat")
async def rwbt_director_ui_chat(request):
    ensure_director_running()
    payload = await request.json()
    response = director_chat(
        session_id=str(payload.get("session_id") or "rwbt-main"),
        user_message=str(payload.get("user_message") or ""),
        system_prompt=str(payload.get("system_prompt") or ""),
        persist_context=bool(payload.get("persist_context", True)),
        model=str(payload.get("model") or ""),
        max_tokens=int(payload.get("max_tokens") or 1200),
        temperature=float(payload.get("temperature") or 0.2),
    )
    return web.json_response(response)


@PromptServer.instance.routes.get("/rwbt_director_ui/jobs")
async def rwbt_director_ui_jobs(request):
    output_root = str(request.rel_url.query.get("output_root") or "")
    return web.json_response({"jobs": list_rwbt_jobs(output_root=output_root)})


@PromptServer.instance.routes.get("/rwbt_director_ui/job_outputs")
async def rwbt_director_ui_job_outputs(request):
    job_id = str(request.rel_url.query.get("job_id") or "").strip()
    output_root = str(request.rel_url.query.get("output_root") or "")
    limit = int(request.rel_url.query.get("limit") or 40)
    if not job_id:
        return web.json_response({"error": "job_id is required"}, status=400)
    return web.json_response(list_job_outputs(job_id=job_id, output_root=output_root, limit=limit))


@PromptServer.instance.routes.get("/rwbt_director_ui/image")
async def rwbt_director_ui_image(request):
    image_path = str(request.rel_url.query.get("path") or "")
    output_root = str(request.rel_url.query.get("output_root") or "")
    resolved = resolve_preview_image_path(image_path=image_path, output_root=output_root)
    if not resolved:
        return web.json_response({"error": "Image not found or not allowed"}, status=404)
    return web.FileResponse(path=resolved, headers={"Content-Type": preview_image_mime(resolved)})


print("[RWBTDirector] Loaded nodes, routes, and web extension")
ensure_director_running()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
