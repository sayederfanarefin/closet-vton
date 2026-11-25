#!/usr/bin/env python3
"""
Check whether a ComfyUI instance can validate a workflow that references model filenames.

This catches "Value not in list" issues early (e.g., missing models in /opt/ComfyUI/models/*).

Usage:
  python scripts/check_comfy_assets.py --workflow workflow_api.json --comfy http://localhost:8188
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _load_workflow(path: Path) -> Dict[str, Any]:
    wf = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(wf, dict):
        raise ValueError("Workflow JSON must be an object mapping node_id -> node")
    return wf


def _get_required_spec(obj_info: Dict[str, Any], class_type: str, input_name: str) -> Optional[Any]:
    node = obj_info.get(class_type)
    if not isinstance(node, dict):
        return None
    required = (node.get("input") or {}).get("required")
    if not isinstance(required, dict):
        return None
    return required.get(input_name)


def _choices_from_spec(spec: Any) -> Optional[List[Any]]:
    # Usually: [[choice1, choice2, ...], {...}] or [[choice1,...]]
    if not isinstance(spec, list) or len(spec) == 0:
        return None
    first = spec[0]
    if isinstance(first, list):
        return first
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Check workflow model dropdown values against a running ComfyUI instance.")
    p.add_argument("--workflow", default="workflow_api.json", help="Path to ComfyUI workflow JSON (default: workflow_api.json)")
    p.add_argument("--comfy", default="http://localhost:8188", help="ComfyUI HTTP base URL (default: http://localhost:8188)")
    args = p.parse_args()

    wf_path = Path(args.workflow).resolve()
    comfy = args.comfy.rstrip("/")

    wf = _load_workflow(wf_path)

    r = requests.get(f"{comfy}/object_info", timeout=10)
    r.raise_for_status()
    obj = r.json()
    if not isinstance(obj, dict):
        raise RuntimeError("Unexpected /object_info response type")

    issues: List[Tuple[str, str, str, str, int]] = []
    # (node_id, class_type, input_name, received_value, choices_len)

    for node_id, node in wf.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if not isinstance(class_type, str):
            continue
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue

        for input_name, received in inputs.items():
            if not isinstance(received, str):
                continue
            spec = _get_required_spec(obj, class_type, str(input_name))
            if spec is None:
                continue
            choices = _choices_from_spec(spec)
            if choices is None:
                continue
            if received not in choices:
                issues.append((str(node_id), class_type, str(input_name), received, len(choices)))

    print(f"Workflow: {wf_path}")
    print(f"ComfyUI:  {comfy}")
    print(f"Issues:   {len(issues)}")
    if issues:
        print("\nInvalid dropdown values (usually missing model files):")
        for node_id, ct, input_name, received, n_choices in issues:
            print(f"- node {node_id} ({ct}) input '{input_name}': '{received}' not in list (len={n_choices})")


if __name__ == "__main__":
    main()

