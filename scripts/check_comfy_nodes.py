#!/usr/bin/env python3
"""
Check whether the ComfyUI instance has all node types required by a workflow JSON.

Usage:
  python scripts/check_comfy_nodes.py --workflow workflow_api.json --comfy http://localhost:8188

This hits ComfyUI's /object_info endpoint and compares it against the workflow's
node "class_type" values.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests


def _load_workflow(path: Path) -> Dict[str, dict]:
    wf = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(wf, dict):
        raise ValueError("Workflow JSON must be an object mapping node_id -> node")
    return wf


def _required_class_types(wf: Dict[str, dict]) -> Tuple[Set[str], Dict[str, List[str]]]:
    required: Set[str] = set()
    ids_by_type: Dict[str, List[str]] = defaultdict(list)
    for node_id, node in wf.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type")
        if not isinstance(ct, str) or not ct:
            continue
        required.add(ct)
        ids_by_type[ct].append(str(node_id))
    return required, ids_by_type


def main() -> None:
    p = argparse.ArgumentParser(description="Check required workflow node types against a running ComfyUI instance.")
    p.add_argument("--workflow", default="workflow_api.json", help="Path to ComfyUI workflow JSON (default: workflow_api.json)")
    p.add_argument("--comfy", default="http://localhost:8188", help="ComfyUI HTTP base URL (default: http://localhost:8188)")
    args = p.parse_args()

    wf_path = Path(args.workflow).resolve()
    comfy = args.comfy.rstrip("/")

    wf = _load_workflow(wf_path)
    required, ids_by_type = _required_class_types(wf)

    r = requests.get(f"{comfy}/object_info", timeout=10)
    r.raise_for_status()
    obj = r.json()
    if not isinstance(obj, dict):
        raise RuntimeError("Unexpected /object_info response type")

    available = set(obj.keys())
    missing = sorted([t for t in required if t not in available])
    extra = sorted([t for t in available if t not in required])

    print(f"Workflow: {wf_path}")
    print(f"ComfyUI:  {comfy}")
    print(f"Required node types: {len(required)}")
    print(f"Available node types: {len(available)}")
    print(f"Missing node types:  {len(missing)}")

    if missing:
        print("\nMissing:")
        for t in missing:
            ids = ",".join(ids_by_type.get(t, []))
            print(f"- {t} (node ids: {ids})")

    # Keep output short; extra can be huge.
    if not missing:
        print("\nAll required node types are available.")


if __name__ == "__main__":
    main()

