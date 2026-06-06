#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply tracked longform Yvann template to base workflow paths")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]), help="ComfyUI repo root")
    parser.add_argument("--apply-user-workflow", action="store_true", help="Also copy template into user/default/workflows")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source = repo_root / "script_examples" / "workflows" / "AudioReactive_ImagesToVideo_Yvann (Longform Launcher).json"
    target_base = repo_root / "custom_nodes" / "comfyui_yvann-nodes" / "example_workflows" / "AudioReactive_ImagesToVideo_Yvann.json"
    user_target = repo_root / "user" / "default" / "workflows" / "AudioReactive_ImagesToVideo_Yvann.json"

    if not source.exists():
        raise FileNotFoundError(f"Source template not found: {source}")
    if not target_base.parent.exists():
        raise FileNotFoundError(f"Yvann workflow path not found: {target_base.parent}")

    target_base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_base)
    print(f"Applied base template: {target_base}")

    if args.apply_user_workflow:
        user_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, user_target)
        print(f"Applied user workflow template: {user_target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
