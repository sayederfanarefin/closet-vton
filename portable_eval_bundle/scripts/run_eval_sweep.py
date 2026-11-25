#!/usr/bin/env python3
"""Run portable eval (overlap + deepfake + fidelity) across all existing runs.

This script scans `<repo_root>/runs/` for run folders that contain:
  - meta.json
  - results.jsonl

It then compares each run to a matching baseline run (same manifest SHA) when
available, otherwise falls back to a chosen baseline if configured.

Outputs:
  <repo_root>/runs/eval_sweep_YYYYMMDD_HHMMSS/
    overlap_<comparison>.csv
    <comparison>/
      deepfake/*
      fidelity/*
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunInfo:
    dir_name: str
    path: Path
    run_id: str
    run_name: str
    manifest_sha256: str
    manifest_path: str
    created_at: Optional[dt.datetime]


def _parse_dt(s: str) -> Optional[dt.datetime]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_run_dirs(runs_root: Path) -> Iterable[Path]:
    for p in sorted(runs_root.iterdir()):
        if not p.is_dir():
            continue
        meta = p / "meta.json"
        results = p / "results.jsonl"
        if meta.exists() and results.exists():
            yield p


def _load_run_info(run_dir: Path) -> RunInfo:
    meta = _read_json(run_dir / "meta.json")
    return RunInfo(
        dir_name=run_dir.name,
        path=run_dir,
        run_id=str(meta.get("run_id", run_dir.name.split("_", 3)[0])),
        run_name=str(meta.get("run_name", run_dir.name)),
        manifest_sha256=str(meta.get("manifest_sha256", "")),
        manifest_path=str(meta.get("manifest_path", "")),
        created_at=_parse_dt(str(meta.get("created_at", ""))),
    )


def _resolve_run_selector(runs_root: Path, selector: str) -> RunInfo:
    direct = runs_root / selector
    if direct.is_dir() and (direct / "meta.json").exists():
        return _load_run_info(direct)

    matches: List[RunInfo] = []
    for run_dir in _iter_run_dirs(runs_root):
        info = _load_run_info(run_dir)
        if selector in (info.run_id, info.run_name, info.dir_name):
            matches.append(info)

    if not matches:
        raise ValueError(f"Could not resolve run selector: {selector}")

    def _sort_key(r: RunInfo) -> Tuple[int, str]:
        # Prefer newest if created_at is available.
        if r.created_at is None:
            return (0, r.dir_name)
        return (1, r.created_at.isoformat())

    matches.sort(key=_sort_key, reverse=True)
    return matches[0]


def _is_baseline(run: RunInfo, baseline_substrs: Sequence[str]) -> bool:
    name = run.run_name.lower()
    return any(s.lower() in name for s in baseline_substrs)


def _pick_manifest_baselines(runs: Sequence[RunInfo], baseline_substrs: Sequence[str]) -> Dict[str, RunInfo]:
    by_manifest: Dict[str, List[RunInfo]] = {}
    for r in runs:
        if not r.manifest_sha256:
            continue
        if _is_baseline(r, baseline_substrs):
            by_manifest.setdefault(r.manifest_sha256, []).append(r)

    chosen: Dict[str, RunInfo] = {}
    for sha, items in by_manifest.items():
        # pick newest baseline for that manifest
        items_sorted = sorted(
            items,
            key=lambda x: (x.created_at is not None, x.created_at or dt.datetime.min, x.dir_name),
            reverse=True,
        )
        chosen[sha] = items_sorted[0]
    return chosen


def _pick_default_fallback_baseline(runs: Sequence[RunInfo], baseline_substrs: Sequence[str]) -> Optional[RunInfo]:
    baselines = [r for r in runs if _is_baseline(r, baseline_substrs)]
    if not baselines:
        return None
    # Prefer an "unpaired" baseline if present.
    unpaired = [b for b in baselines if "unpaired" in b.run_name.lower()]
    pool = unpaired if unpaired else baselines
    pool_sorted = sorted(
        pool,
        key=lambda x: (x.created_at is not None, x.created_at or dt.datetime.min, x.dir_name),
        reverse=True,
    )
    return pool_sorted[0]


def _sanitize_slug(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("._-")
    return slug or "comparison"


def _short_baseline_label(run_name: str) -> str:
    if run_name.startswith("vitonhd_test_"):
        return run_name.replace("vitonhd_test_", "", 1)
    return run_name


def _count_csv_rows(path: Path) -> int:
    # Count data rows (excluding header). Works even for large CSVs via buffered IO.
    with path.open("rb") as f:
        count = -1  # header
        for _ in f:
            count += 1
    return max(0, count)


def _run_cmd(cmd: List[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("."), help="Project root (contains runs/).")
    p.add_argument("--runs-root", type=Path, default=None, help="Override runs directory (default: <repo_root>/runs).")
    p.add_argument("--out-root", type=Path, default=None, help="Override output directory.")
    p.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    p.add_argument("--baseline-substr", default="vitonhd_test", help="Comma-separated baseline name substrings.")
    p.add_argument(
        "--fallback-baseline",
        default="auto",
        help="Run selector to use when no matching-manifest baseline exists. Use 'auto' or '' to disable.",
    )
    p.add_argument("--max-samples", type=int, default=0, help="0 means all rows.")
    p.add_argument(
        "--detectors",
        default="fft_peakiness,jpeg_blockiness,noise_inconsistency,chroma_misalignment",
        help="Comma-separated deepfake detector names.",
    )
    p.add_argument("--use-mediapipe", action="store_true", help="Enable MediaPipe face/pose scoring.")
    p.add_argument("--visualize", action="store_true", help="Run visualization at end.")
    p.add_argument("--skip-baselines", action="store_true", help="Skip comparing baseline runs to each other.")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failure (default: continue).")
    p.add_argument("--dry-run", action="store_true", help="Print planned comparisons without running.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    runs_root = (args.runs_root.resolve() if args.runs_root else (repo_root / "runs").resolve())
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    bundle_dir = Path(__file__).resolve().parents[1]
    scripts_dir = bundle_dir / "scripts"
    build_overlap = scripts_dir / "build_overlap_manifest.py"
    run_deepfake = scripts_dir / "run_deepfake_comparison.py"
    run_fidelity = scripts_dir / "run_fidelity_comparison.py"
    viz_script = scripts_dir / "visualize_eval_transfer.py"

    runs = [_load_run_info(p) for p in _iter_run_dirs(runs_root)]
    if not runs:
        raise RuntimeError(f"No run folders (meta.json + results.jsonl) found under: {runs_root}")

    baseline_substrs = [x.strip() for x in args.baseline_substr.split(",") if x.strip()]
    manifest_baselines = _pick_manifest_baselines(runs, baseline_substrs)

    fallback: Optional[RunInfo]
    if args.fallback_baseline == "auto":
        fallback = _pick_default_fallback_baseline(runs, baseline_substrs)
    elif not args.fallback_baseline.strip():
        fallback = None
    else:
        fallback = _resolve_run_selector(runs_root, args.fallback_baseline.strip())

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (args.out_root.resolve() if args.out_root else (repo_root / "runs" / f"eval_sweep_{stamp}").resolve())
    out_root.mkdir(parents=True, exist_ok=True)

    max_args: List[str] = []
    if int(args.max_samples) > 0:
        max_args = ["--max-samples", str(int(args.max_samples))]

    fidelity_extra: List[str] = []
    if bool(args.use_mediapipe):
        fidelity_extra = ["--use-mediapipe-face", "--use-mediapipe-pose"]

    planned: List[Tuple[RunInfo, RunInfo, str, bool]] = []
    used_slugs: set[str] = set()
    for r in runs:
        # Skip baseline self-comparisons.
        if _is_baseline(r, baseline_substrs):
            continue

        base = manifest_baselines.get(r.manifest_sha256)
        used_fallback = False
        if base is None:
            base = fallback
            used_fallback = base is not None
        if base is None:
            print(f"Skip (no baseline found): {r.dir_name}  manifest={r.manifest_sha256[:10]}")
            continue
        if args.skip_baselines and _is_baseline(base, baseline_substrs) and _is_baseline(r, baseline_substrs):
            continue

        base_label = _short_baseline_label(base.run_name)
        run_label = r.run_name
        slug = _sanitize_slug(f"{base_label}_vs_{run_label}")
        if slug in used_slugs:
            slug = _sanitize_slug(f"{base_label}_vs_{run_label}__{r.run_id}")
        used_slugs.add(slug)
        planned.append((base, r, slug, used_fallback))

    if not planned:
        raise RuntimeError("No comparisons planned. Check baseline selection / filters.")

    print(f"Repo root: {repo_root}", flush=True)
    print(f"Runs root: {runs_root}", flush=True)
    print(f"Output root: {out_root}", flush=True)
    print(f"Python: {args.python_bin}", flush=True)
    print(f"Comparisons planned: {len(planned)}", flush=True)
    print("", flush=True)

    manifest_map_path = out_root / "runs_index.json"
    manifest_map_path.write_text(
        json.dumps(
            {
                "runs_root": str(runs_root),
                "baseline_substrs": baseline_substrs,
                "fallback_baseline": (fallback.dir_name if fallback else ""),
                "runs": [
                    {
                        "dir_name": r.dir_name,
                        "run_id": r.run_id,
                        "run_name": r.run_name,
                        "manifest_sha256": r.manifest_sha256,
                        "manifest_path": r.manifest_path,
                        "created_at": (r.created_at.isoformat() if r.created_at else ""),
                    }
                    for r in runs
                ],
                "planned": [
                    {
                        "baseline_dir": b.dir_name,
                        "run_dir": r.dir_name,
                        "slug": slug,
                        "used_fallback": used_fallback,
                    }
                    for (b, r, slug, used_fallback) in planned
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"Dry run. Planned comparisons written to: {manifest_map_path}", flush=True)
        return 0

    failures: List[str] = []

    for idx, (base, run, slug, used_fallback) in enumerate(planned, start=1):
        print("", flush=True)
        print(f"[{idx}/{len(planned)}] {slug}", flush=True)
        if used_fallback and base.manifest_sha256 and run.manifest_sha256 and base.manifest_sha256 != run.manifest_sha256:
            print("  NOTE: baseline manifest != run manifest (fallback baseline in use).", flush=True)

        overlap_csv = out_root / f"overlap_{slug}.csv"
        comp_dir = out_root / slug
        deepfake_out = comp_dir / "deepfake"
        fidelity_out = comp_dir / "fidelity"
        comp_dir.mkdir(parents=True, exist_ok=True)

        try:
            _run_cmd(
                [
                    args.python_bin,
                    str(build_overlap),
                    "--runs-root",
                    str(runs_root),
                    "--run-a",
                    base.dir_name,
                    "--run-b",
                    run.dir_name,
                    "--out",
                    str(overlap_csv),
                ],
                cwd=repo_root,
            )
        except subprocess.CalledProcessError as e:
            msg = f"{slug}: build_overlap_manifest failed (exit={e.returncode})"
            failures.append(msg)
            print(f"  ERROR: {msg}", flush=True)
            if args.fail_fast:
                raise
            continue

        overlap_rows = _count_csv_rows(overlap_csv)
        print(f"  overlap_rows: {overlap_rows}", flush=True)
        if overlap_rows <= 0:
            print("  Skip: no overlap.", flush=True)
            continue

        try:
            _run_cmd(
                [
                    args.python_bin,
                    str(run_deepfake),
                    "--overlap-csv",
                    str(overlap_csv),
                    "--outdir",
                    str(deepfake_out),
                    "--detectors",
                    str(args.detectors),
                    *max_args,
                ],
                cwd=repo_root,
            )
        except subprocess.CalledProcessError as e:
            msg = f"{slug}: run_deepfake_comparison failed (exit={e.returncode})"
            failures.append(msg)
            print(f"  ERROR: {msg}", flush=True)
            if args.fail_fast:
                raise
            continue

        try:
            _run_cmd(
                [
                    args.python_bin,
                    str(run_fidelity),
                    "--overlap-csv",
                    str(overlap_csv),
                    "--outdir",
                    str(fidelity_out),
                    *max_args,
                    *fidelity_extra,
                ],
                cwd=repo_root,
            )
        except subprocess.CalledProcessError as e:
            msg = f"{slug}: run_fidelity_comparison failed (exit={e.returncode})"
            failures.append(msg)
            print(f"  ERROR: {msg}", flush=True)
            if args.fail_fast:
                raise
            continue

    if args.visualize:
        try:
            _run_cmd(
                [
                    args.python_bin,
                    str(viz_script),
                    "--eval-root",
                    str(out_root),
                ],
                cwd=repo_root,
            )
        except subprocess.CalledProcessError as e:
            msg = f"visualize_eval_transfer failed (exit={e.returncode})"
            failures.append(msg)
            print(f"ERROR: {msg}", flush=True)
            if args.fail_fast:
                raise

    if failures:
        (out_root / "sweep_errors.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")

    print("", flush=True)
    print("Done.", flush=True)
    print(f"All outputs are in: {out_root}", flush=True)
    if failures:
        print(f"Failures: {len(failures)} (see {out_root / 'sweep_errors.txt'})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
