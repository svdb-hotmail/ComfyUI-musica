#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd, cwd=None, check=True):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        shell=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc


def load_manifest(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("required_nodes", [])
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Manifest must contain a non-empty 'required_nodes' list")
    for node in nodes:
        if not node.get("name") or not node.get("repo"):
            raise ValueError("Each node entry requires 'name' and 'repo'")
    return nodes


def git_pull(repo_dir: Path):
    run(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
    run(["git", "pull", "--ff-only"], cwd=repo_dir)


def git_clone(target_dir: Path, repo: str, branch: str | None):
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([repo, str(target_dir)])
    run(cmd)


def install_requirements(python_bin: str, node_dir: Path):
    req = node_dir / "requirements.txt"
    if not req.exists():
        return "no_requirements"
    run([python_bin, "-m", "pip", "install", "-r", str(req)])
    return "requirements_installed"


def get_head_sha(node_dir: Path):
    git_dir = node_dir / ".git"
    if not git_dir.exists():
        return None
    proc = run(["git", "rev-parse", "HEAD"], cwd=node_dir, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def main():
    parser = argparse.ArgumentParser(
        description="Sync required ComfyUI custom nodes for Yvann workflows"
    )
    parser.add_argument(
        "--comfy-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="ComfyUI root directory (defaults to this repo root)",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parent / "required_custom_nodes.json"),
        help="Path to manifest JSON",
    )
    parser.add_argument(
        "--install-requirements",
        action="store_true",
        help="Install requirements.txt for each synced node",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable for pip installs",
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "verify"],
        default="sync",
        help="sync clones/updates repos; verify only checks presence",
    )
    args = parser.parse_args()

    comfy_root = Path(args.comfy_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    custom_nodes_dir = comfy_root / "custom_nodes"

    if not custom_nodes_dir.exists():
        raise FileNotFoundError(f"custom_nodes not found at {custom_nodes_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    nodes = load_manifest(manifest_path)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "comfy_root": str(comfy_root),
        "manifest": str(manifest_path),
        "mode": args.mode,
        "install_requirements": args.install_requirements,
        "results": [],
    }

    for node in nodes:
        name = node["name"]
        repo = node["repo"]
        branch = node.get("branch")
        target = custom_nodes_dir / name
        result = {
            "name": name,
            "repo": repo,
            "branch": branch,
            "path": str(target),
            "status": "unknown",
            "requirements": "skipped",
            "head": None,
            "error": None,
        }

        try:
            if not target.exists():
                if args.mode == "verify":
                    result["status"] = "missing"
                else:
                    git_clone(target, repo, branch)
                    result["status"] = "cloned"
            else:
                if (target / ".git").exists():
                    if args.mode == "sync":
                        git_pull(target)
                        result["status"] = "updated"
                    else:
                        result["status"] = "present_git"
                else:
                    result["status"] = "present_non_git"

            if args.install_requirements and result["status"] not in {"missing"}:
                result["requirements"] = install_requirements(args.python_bin, target)

            result["head"] = get_head_sha(target)
        except Exception as exc:  # noqa: BLE001
            result["status"] = "error"
            result["error"] = str(exc)

        report["results"].append(result)

    out_path = custom_nodes_dir / "required_nodes_sync_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote report: {out_path}")
    failures = [r for r in report["results"] if r["status"] in {"missing", "error"}]
    if failures:
        print(f"Sync finished with {len(failures)} issue(s).")
        return 1
    print("Sync finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
