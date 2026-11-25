#!/usr/bin/env python3
"""Build a fair overlap manifest for pairwise VTON run comparison.

The script reads `results.jsonl` for two runs and matches only samples that
share the same `(cloth_id, person_id)` pair and have `status == "ok"` in both.
It writes a CSV that can be consumed by downstream evaluators (deepfake
detectors, cloth/face/pose scoring, etc.).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class Sample:
    run_dir: str
    run_name: str
    run_id: str
    pair_key: Tuple[str, str]  # (cloth_id, person_id)
    cloth_id: str
    person_id: str
    sample_id: str
    pred_path: str
    person_path: str
    cloth_path: str
    mask_path: str
    gt_path: str


def _basename_no_ext(path_str: str) -> str:
    return Path(path_str).stem


def _localize_path(path_str: Optional[str], repo_root: Path) -> str:
    """Best-effort remap of stale absolute paths into current repo root."""
    if not path_str:
        return ""

    normalized = path_str.replace("\\", "/")
    for marker in ("/data/", "/runs/"):
        idx = normalized.find(marker)
        if idx != -1:
            rel = normalized[idx + 1 :]  # drop leading slash
            candidate = repo_root / rel
            if candidate.exists():
                return str(candidate)

    p = Path(path_str)
    if p.is_absolute():
        # Avoid slow existence checks for stale absolute paths outside repo root.
        try:
            p.relative_to(repo_root)
        except ValueError:
            return path_str
        return str(p) if p.exists() else path_str

    candidate = repo_root / p
    if candidate.exists():
        return str(candidate)
    if p.exists():
        return str(p)
    return path_str


def _extract_pair_key(row: dict) -> Tuple[str, str]:
    sample_id = row.get("id", "")
    if "__" in sample_id:
        cloth_id, person_id = sample_id.split("__", 1)
        return cloth_id, person_id

    cloth_path = row.get("cloth_path") or ""
    person_path = row.get("person_path") or ""
    cloth_id = _basename_no_ext(cloth_path)
    person_id = _basename_no_ext(person_path)
    return cloth_id, person_id


def _iter_ok_samples(run_dir: Path, repo_root: Path) -> Iterable[Sample]:
    meta_path = run_dir / "meta.json"
    results_path = run_dir / "results.jsonl"
    if not meta_path.exists() or not results_path.exists():
        return

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    run_name = meta.get("run_name", run_dir.name)
    run_id = meta.get("run_id", run_dir.name.split("_", 3)[0])

    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue

            cloth_id, person_id = _extract_pair_key(row)
            yield Sample(
                run_dir=run_dir.name,
                run_name=run_name,
                run_id=run_id,
                pair_key=(cloth_id, person_id),
                cloth_id=cloth_id,
                person_id=person_id,
                sample_id=row.get("id", ""),
                pred_path=_localize_path(row.get("pred_path"), repo_root),
                person_path=_localize_path(row.get("person_path"), repo_root),
                cloth_path=_localize_path(row.get("cloth_path"), repo_root),
                mask_path=_localize_path(row.get("mask_path"), repo_root),
                gt_path=_localize_path(row.get("gt_path"), repo_root),
            )


def _load_run_index(run_dir: Path, repo_root: Path) -> Dict[Tuple[str, str], Sample]:
    index: Dict[Tuple[str, str], Sample] = {}
    for sample in _iter_ok_samples(run_dir, repo_root):
        index[sample.pair_key] = sample
    return index


def _resolve_run_dir(runs_root: Path, selector: str) -> Path:
    direct = runs_root / selector
    if direct.is_dir():
        return direct

    matches = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        run_name = meta.get("run_name", "")
        run_id = meta.get("run_id", "")
        if selector in (run_name, run_id):
            matches.append(run_dir)

    if not matches:
        raise ValueError(f"Could not resolve run selector: {selector}")
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise ValueError(f"Selector '{selector}' is ambiguous. Matches: {names}")
    return matches[0]


def _write_overlap_csv(
    out_csv: Path,
    a_name: str,
    b_name: str,
    index_a: Dict[Tuple[str, str], Sample],
    index_b: Dict[Tuple[str, str], Sample],
) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    overlap_keys = sorted(set(index_a.keys()) & set(index_b.keys()))

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "cloth_id",
                "person_id",
                "sample_id_a",
                "sample_id_b",
                f"pred_path_{a_name}",
                f"pred_path_{b_name}",
                "person_path",
                f"cloth_path_{a_name}",
                f"cloth_path_{b_name}",
                "mask_path",
                f"gt_path_{a_name}",
                f"gt_path_{b_name}",
            ],
        )
        writer.writeheader()
        for key in overlap_keys:
            sa = index_a[key]
            sb = index_b[key]
            writer.writerow(
                {
                    "pair_id": f"{sa.cloth_id}__{sa.person_id}",
                    "cloth_id": sa.cloth_id,
                    "person_id": sa.person_id,
                    "sample_id_a": sa.sample_id,
                    "sample_id_b": sb.sample_id,
                    f"pred_path_{a_name}": sa.pred_path,
                    f"pred_path_{b_name}": sb.pred_path,
                    "person_path": sa.person_path or sb.person_path,
                    f"cloth_path_{a_name}": sa.cloth_path,
                    f"cloth_path_{b_name}": sb.cloth_path,
                    "mask_path": sa.mask_path or sb.mask_path,
                    f"gt_path_{a_name}": sa.gt_path,
                    f"gt_path_{b_name}": sb.gt_path,
                }
            )
    return len(overlap_keys)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Directory that contains run folders (default: runs).",
    )
    parser.add_argument(
        "--run-a",
        required=True,
        help="Run folder name, run_id, or run_name for side A.",
    )
    parser.add_argument(
        "--run-b",
        required=True,
        help="Run folder name, run_id, or run_name for side B.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/overlap_manifest.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs_root = args.runs_root.resolve()
    repo_root = runs_root.parent

    run_dir_a = _resolve_run_dir(runs_root, args.run_a)
    run_dir_b = _resolve_run_dir(runs_root, args.run_b)

    index_a = _load_run_index(run_dir_a, repo_root)
    index_b = _load_run_index(run_dir_b, repo_root)

    overlap_count = _write_overlap_csv(
        out_csv=args.out,
        a_name=run_dir_a.name,
        b_name=run_dir_b.name,
        index_a=index_a,
        index_b=index_b,
    )

    print(f"run_a: {run_dir_a.name} ok_pairs={len(index_a)}")
    print(f"run_b: {run_dir_b.name} ok_pairs={len(index_b)}")
    print(f"overlap_pairs: {overlap_count}")
    print(f"output_csv: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
