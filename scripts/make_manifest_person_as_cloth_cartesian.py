#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

# Allow importing the local `evaluation/` package when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.manifest import Sample, load_manifest  # noqa: E402


def _relpath(path: Path, base_dir: Path, *, relative: bool) -> str:
    if not relative:
        return str(path)
    return os.path.relpath(str(path), start=str(base_dir))


def _to_entry(sample: Sample, *, out_base_dir: Path, relative: bool) -> Dict[str, Any]:
    return {
        "id": sample.sample_id,
        "person": _relpath(sample.person_path, out_base_dir, relative=relative),
        "cloth": _relpath(sample.cloth_path, out_base_dir, relative=relative),
        "mask": _relpath(sample.mask_path, out_base_dir, relative=relative) if sample.mask_path else None,
        "category": sample.category or None,
        "meta": sample.meta or {},
    }


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Create a cartesian-product manifest where the required 'cloth' input is a PERSON image. "
            "For each pair (A,B), we set: cloth=A.person_image, person=B.person_image."
        )
    )
    p.add_argument(
        "--base-manifest",
        default="data/viton-hd/manifest_test_paired.jsonl",
        help="Base manifest to source person<->cloth mappings (default: VITON-HD test paired).",
    )
    p.add_argument(
        "--out",
        default="data/viton-hd/manifest_test_person_as_cloth_cartesian.jsonl",
        help="Output manifest path (.jsonl or .csv).",
    )
    p.add_argument(
        "--include-self",
        action="store_true",
        help="Include pairs where cloth-person id == person id (default: exclude).",
    )
    p.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write paths relative to the output manifest directory (recommended; smaller + portable).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, stop after writing N rows (useful for debugging).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=250000,
        help="Print progress every N rows (default: 250k).",
    )
    args = p.parse_args()

    base_manifest_path = Path(args.base_manifest).resolve()
    out_path = Path(args.out).resolve()
    out_base_dir = out_path.parent

    base_samples = load_manifest(str(base_manifest_path))
    if not base_samples:
        raise SystemExit(f"No samples found in base manifest: {base_manifest_path}")

    entries = [_to_entry(s, out_base_dir=out_base_dir, relative=args.relative_paths) for s in base_samples]
    n = len(entries)
    total = n * n if args.include_self else n * (n - 1)

    print(f"[make_manifest_person_as_cloth_cartesian] base_samples={n} -> rows={total} (include_self={args.include_self})")
    print(f"[make_manifest_person_as_cloth_cartesian] out={out_path} (relative_paths={args.relative_paths})")

    rows_written = 0

    def row_iter():
        nonlocal rows_written
        for a in entries:
            a_id = str(a["id"])
            a_person_img = str(a["person"])
            a_person_cloth = str(a["cloth"])  # garment image associated with A (requested for validation)
            for b in entries:
                b_id = str(b["id"])
                if (not args.include_self) and (a_id == b_id):
                    continue

                out_row: Dict[str, Any] = {
                    "id": f"{a_id}__{b_id}",
                    # Required fields consumed by evaluation runner:
                    "person": str(b["person"]),
                    "cloth": a_person_img,  # NOTE: cloth input is a PERSON image (A)
                    "gt": None,
                    "mask": b.get("mask"),
                    "category": b.get("category"),
                    # Extra fields for verification/validation:
                    "c_id": a_id,
                    "p_id": b_id,
                    "c_cloth": a_person_cloth,
                }

                rows_written += 1
                if args.progress_every and (rows_written % args.progress_every == 0):
                    print(f"[make_manifest_person_as_cloth_cartesian] wrote {rows_written}/{total}")

                yield out_row

                if args.max_rows and rows_written >= args.max_rows:
                    return

    # Write streaming to avoid holding all cartesian rows in memory.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        # CSV is much smaller than JSONL for very large manifests (no repeated keys).
        fieldnames = ["id", "person", "cloth", "gt", "mask", "category", "c_id", "p_id", "c_cloth"]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in row_iter():
                w.writerow({k: row.get(k) for k in fieldnames})
    else:
        with out_path.open("w", encoding="utf-8") as f:
            for row in row_iter():
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"[make_manifest_person_as_cloth_cartesian] done: wrote {rows_written} rows")


if __name__ == "__main__":
    main()
