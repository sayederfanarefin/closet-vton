#!/usr/bin/env python3
"""
Install (or update) the ComfyUI custom node packs required by workflow_api.json.

Run this with the same Python environment ComfyUI uses.

Examples:
  # Local ComfyUI (venv activated):
  python scripts/install_comfy_custom_nodes.py --comfy-dir /path/to/ComfyUI --update

  # Inside the Docker container:
  python3 /app/scripts/install_comfy_custom_nodes.py --comfy-dir /opt/ComfyUI --update
"""

import argparse
import subprocess
import shutil
import sys
from pathlib import Path


REPOS = [
    # Required by workflow_api.json
    ("Comfyui-In-Context-Lora-Utils", "https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils.git"),
    ("ComfyUI-Inpaint-Nodes", "https://github.com/acly/ComfyUI-Inpaint-Nodes.git"),
    ("ComfyUI_essentials", "https://github.com/cubiq/ComfyUI_essentials.git"),
    ("ComfyUI-GGUF", "https://github.com/city96/ComfyUI-GGUF.git"),
    ("comfyui_segment_anything", "https://github.com/storyicon/comfyui_segment_anything.git"),
]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def main() -> None:
    p = argparse.ArgumentParser(description="Install ComfyUI custom node packs required by workflow_api.json.")
    p.add_argument("--comfy-dir", default="/opt/ComfyUI", help="Path to ComfyUI root (default: /opt/ComfyUI).")
    p.add_argument("--update", action="store_true", help="If repo exists, git pull instead of skipping.")
    args = p.parse_args()

    comfy_dir = Path(args.comfy_dir).expanduser().resolve()
    custom_nodes = comfy_dir / "custom_nodes"
    if not comfy_dir.exists():
        raise SystemExit(f"ComfyUI dir not found: {comfy_dir}")
    custom_nodes.mkdir(parents=True, exist_ok=True)

    for name, url in REPOS:
        dest = custom_nodes / name
        if dest.exists():
            if args.update and (dest / ".git").exists():
                _run(["git", "pull", "--ff-only"], cwd=dest)
            else:
                print(f"- exists: {dest}")
        else:
            _run(["git", "clone", "--depth", "1", url, str(dest)])

        req = dest / "requirements.txt"
        if req.exists():
            _run([sys.executable, "-m", "pip", "install", "-r", str(req)])

    # Repository-owned fail-closed composition and final person validation nodes.
    local_guard = Path(__file__).resolve().parents[1] / "comfy_custom_nodes" / "crop_integrity"
    shutil.copytree(local_guard, custom_nodes / "crop_integrity", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # comfyui_segment_anything expects Meta's Segment Anything python module.
    _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "git+https://github.com/facebookresearch/segment-anything.git"])

    # Some GGUF node packs expect the gguf python package.
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "gguf"])

    print("\nDone. Restart ComfyUI, then re-check:")
    print("  python scripts/check_comfy_nodes.py --workflow workflow_api.json --comfy http://localhost:8188")


if __name__ == "__main__":
    main()
