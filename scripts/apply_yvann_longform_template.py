#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply tracked longform Yvann dashboard workflow to user workflow paths")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]), help="ComfyUI repo root")
    parser.add_argument("--apply-user-workflow", action="store_true", help="Also copy template into user/default/workflows")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source = repo_root / "script_examples" / "workflows" / "AudioReactive_ImagesToVideo_Yvann (Longform Launcher).json"
    user_target = repo_root / "user" / "default" / "workflows" / "AudioReactive_ImagesToVideo_Yvann.json"

    if not source.exists():
        raise FileNotFoundError(f"Source template not found: {source}")
    if args.apply_user_workflow:
        user_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, user_target)
        sys.stdout.write(f"Applied dashboard workflow: {user_target}\n")
    else:
        sys.stdout.write(f"Dashboard source verified: {source}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
