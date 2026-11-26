import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from evaluation.io_utils import (
    json_dumps,
    load_image_rgb,
    load_mask,
    resize_rgb_tensor,
    save_comparison_grid,
    sha256_file,
)
from evaluation.manifest import load_manifest
from evaluation.metrics import compute_metrics


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "run"


def _safe_filename(s: str) -> str:
    s = s.replace(os.sep, "_").replace("/", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("._") or "sample"


def _git_info(repo_dir: Path) -> Dict[str, Any]:
    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(args, cwd=str(repo_dir), stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    short = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "short_commit": short,
        "dirty": bool(dirty),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=True) + "\n")


def _aggregate(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    t = torch.tensor(values, dtype=torch.float32)
    return {"mean": float(t.mean().item()), "std": float(t.std(unbiased=False).item())}

def _materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink() and not dst.exists():
        dst.unlink(missing_ok=True)

    if dst.exists() or dst.is_symlink():
        return

    if mode == "symlink":
        try:
            target = os.path.relpath(str(src), start=str(dst.parent))
            os.symlink(target, str(dst))
            return
        except Exception:
            pass
    elif mode == "hardlink":
        try:
            os.link(str(src), str(dst))
            return
        except Exception:
            pass
    elif mode != "copy":
        raise ValueError(f"Unknown triplet materialization mode: {mode}")

    shutil.copy2(src, dst)

def _save_triplet(
    *,
    triplets_root: Path,
    safe_id: str,
    person_src: Path,
    cloth_src: Path,
    pred_src: Path,
    mode: str,
) -> Dict[str, str]:
    sample_dir = triplets_root / safe_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    person_ext = person_src.suffix.lower() if person_src.suffix else ".png"
    cloth_ext = cloth_src.suffix.lower() if cloth_src.suffix else ".png"
    pred_ext = pred_src.suffix.lower() if pred_src.suffix else ".png"

    person_dst = sample_dir / f"person{person_ext}"
    cloth_dst = sample_dir / f"cloth{cloth_ext}"
    pred_dst = sample_dir / f"pred{pred_ext}"

    _materialize_file(person_src, person_dst, mode=mode)
    _materialize_file(cloth_src, cloth_dst, mode=mode)
    _materialize_file(pred_src, pred_dst, mode=mode)

    return {
        "triplet_dir": str(sample_dir),
        "triplet_person": str(person_dst),
        "triplet_cloth": str(cloth_dst),
        "triplet_pred": str(pred_dst),
    }

def _append_run_index_row(index_path: Path, header: List[str], row: List[str]) -> None:
    # Allow parallel runs without corrupting the index.
    try:
        import fcntl  # type: ignore
    except Exception:
        fcntl = None

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a+", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            is_empty = f.tell() == 0
            if is_empty:
                f.write(",".join(header) + "\n")
            f.write(",".join(row) + "\n")
            f.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _find_pred(pred_dir: Path, sample_id: str, exts: List[str]) -> Optional[Path]:
    safe_id = _safe_filename(sample_id)
    for ext in exts:
        p = pred_dir / f"{safe_id}{ext}"
        if p.exists():
            return p
    # Fallback: try any extension
    any_match = list(pred_dir.glob(f"{safe_id}.*"))
    return any_match[0] if any_match else None


def main() -> None:
    p = argparse.ArgumentParser(description="Score a folder of predicted images against a manifest.")
    p.add_argument("--manifest", required=True, help="Path to dataset manifest (.jsonl/.json/.csv).")
    p.add_argument("--pred-dir", required=True, help="Directory containing predictions named <id>.<ext>.")
    p.add_argument("--outdir", default="runs", help="Root directory for run artifacts (default: runs).")
    p.add_argument("--run-name", default="", help="Optional label for the run.")
    p.add_argument("--pred-exts", default=".png,.jpg,.jpeg,.webp", help="Comma-separated extensions to try.")
    p.add_argument("--max-samples", type=int, default=0, help="If >0, limit to first N samples.")
    p.add_argument("--resize-to-gt", action="store_true", help="Resize prediction to GT size before metrics (recommended).")
    p.add_argument("--no-metrics", action="store_true", help="Skip reference metrics (PSNR/SSIM); skips reconstruction scoring.")
    p.add_argument("--lpips", action="store_true", help="Compute paired LPIPS perceptual distance (lower is better; requires GT).")
    p.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default=os.getenv("LPIPS_NET", "alex"))
    p.add_argument("--lpips-device", default=os.getenv("LPIPS_DEVICE", ""), help="LPIPS device (e.g. cuda or cpu). Default: auto.")
    p.add_argument("--save-comparisons", action="store_true", help="Save 2x2 grids (person/cloth/pred/gt).")
    p.add_argument(
        "--copy-preds",
        action="store_true",
        help="Copy baseline predictions into the run folder (runs/<...>/images/pred) for reproducible versioned artifacts.",
    )
    p.add_argument("--save-triplets", action="store_true", help="Save per-sample triplets (person/cloth/pred) under runs/<...>/triplets/<id>/")
    p.add_argument(
        "--triplets-mode",
        choices=["symlink", "hardlink", "copy"],
        default=os.getenv("EVAL_TRIPLETS_MODE", "symlink"),
        help="How to materialize triplet files: symlink (default), hardlink, or copy.",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar / ETA output.")


    args = p.parse_args()

    repo_dir = Path(__file__).resolve().parent.parent
    out_root = Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    samples = load_manifest(str(manifest_path))
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(3).hex()
    run_name = _safe_slug(args.run_name) if args.run_name else "score"
    run_dir = out_root / f"{run_id}_{run_name}"

    images_cmp_dir = run_dir / "images" / "compare"
    images_pred_dir = run_dir / "images" / "pred"
    triplets_root = run_dir / "triplets"
    results_path = run_dir / "results.jsonl"

    workflow_path = repo_dir / "workflow_api.json"
    workflow_hash = sha256_file(workflow_path) if workflow_path.exists() else None
    manifest_hash = sha256_file(manifest_path)
    git = _git_info(repo_dir)

    meta = {
        "run_id": run_id,
        "run_name": args.run_name,
        "created_at": _utc_now_iso(),
        "command": " ".join([json.dumps(x) if " " in x else x for x in os.sys.argv]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "pred_dir": str(pred_dir),
        "workflow_sha256": workflow_hash,
        "git": git,
        "env": {
            "python": os.sys.version,
            "torch": getattr(torch, "__version__", None),
        },
        "options": {
            "max_samples": args.max_samples,
            "resize_to_gt": bool(args.resize_to_gt),
            "no_metrics": bool(args.no_metrics),
            "lpips": bool(args.lpips),
            "lpips_net": args.lpips_net,
            "lpips_device": args.lpips_device or None,
            "save_comparisons": bool(args.save_comparisons),
            "copy_preds": bool(args.copy_preds),
            "save_triplets": bool(args.save_triplets),
            "triplets_mode": args.triplets_mode,
        },
    }
    _write_json(run_dir / "meta.json", meta)


    lpips_scorer = None
    if args.lpips:
        if args.no_metrics:
            p.error("--lpips cannot be combined with --no-metrics")
        from evaluation.lpips_metric import LpipsScorer

        lpips_scorer = LpipsScorer(net=args.lpips_net, device=args.lpips_device or None)
        _write_json(
            run_dir / "lpips_info.json",
            {"net": args.lpips_net, "resolved_device": str(lpips_scorer.device)},
        )

    metric_values: Dict[str, List[float]] = {}
    metric_values_by_cat: Dict[str, Dict[str, List[float]]] = {}
    n_success = 0
    n_failed = 0

    exts = [e.strip() for e in args.pred_exts.split(",") if e.strip()]
    pbar = None
    iterable = samples
    if not args.no_progress:
        from tqdm import tqdm

        pbar = tqdm(samples, total=len(samples), desc="score", unit="sample", dynamic_ncols=True)
        iterable = pbar

    for idx, sample in enumerate(iterable, start=1):
        t0 = time.time()
        row: Dict[str, Any] = {
            "id": sample.sample_id,
            "index": idx,
            "person_path": str(sample.person_path),
            "cloth_path": str(sample.cloth_path),
            "gt_path": str(sample.gt_path) if sample.gt_path else None,
            "mask_path": str(sample.mask_path) if sample.mask_path else None,
            "category": sample.category,
        }

        try:
            pred_path = _find_pred(pred_dir, sample.sample_id, exts)
            if pred_path is None:
                raise FileNotFoundError(f"Prediction not found for id={sample.sample_id} in {pred_dir}")

            safe_id = _safe_filename(sample.sample_id)
            if args.copy_preds:
                images_pred_dir.mkdir(parents=True, exist_ok=True)
                snap_path = images_pred_dir / f"{safe_id}{pred_path.suffix.lower()}"
                snap_path.write_bytes(pred_path.read_bytes())
                row["pred_source_path"] = str(pred_path)
                pred_path = snap_path

            pred = load_image_rgb(pred_path)
            gt = load_image_rgb(sample.gt_path) if sample.gt_path else None
            mask = load_mask(sample.mask_path) if sample.mask_path else None

            row_metrics: Dict[str, float] = {}

            if (gt is not None) and args.resize_to_gt:
                pred = resize_rgb_tensor(pred, gt.shape[1], gt.shape[2])

            row["status"] = "ok"
            row["pred_path"] = str(pred_path)

            if (not args.no_metrics) and (gt is not None):
                metrics = compute_metrics(pred=pred, gt=gt, mask=mask)
                row_metrics.update({k: float(v) for k, v in metrics.values.items()})
                if lpips_scorer is not None:
                    row_metrics["lpips"] = lpips_scorer.score(pred, gt)
            elif (not args.no_metrics) and (gt is None):
                row["metrics_note"] = "No ground-truth provided; skipping reference metrics (PSNR/SSIM)."


            row["metrics"] = row_metrics if row_metrics else None
            if row_metrics:
                for k, v in row_metrics.items():
                    metric_values.setdefault(k, []).append(float(v))
                    if sample.category:
                        metric_values_by_cat.setdefault(sample.category, {}).setdefault(k, []).append(float(v))

            if args.save_comparisons:
                person = load_image_rgb(sample.person_path)
                cloth = load_image_rgb(sample.cloth_path)
                save_comparison_grid(
                    person=person,
                    cloth=cloth,
                    pred=pred,
                    gt=gt,
                    path=images_cmp_dir / f"{safe_id}.png",
                )

            if args.save_triplets:
                row["triplet"] = _save_triplet(
                    triplets_root=triplets_root,
                    safe_id=safe_id,
                    person_src=sample.person_path,
                    cloth_src=sample.cloth_path,
                    pred_src=pred_path,
                    mode=args.triplets_mode,
                )

            n_success += 1

        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            n_failed += 1
        finally:
            row["elapsed_sec"] = round(time.time() - t0, 4)
            _append_jsonl(results_path, row)
            if pbar is not None:
                safe_id = _safe_filename(sample.sample_id)
                pbar.set_postfix({"ok": n_success, "err": n_failed, "id": safe_id, "last_s": row["elapsed_sec"]})

    if pbar is not None:
        pbar.close()

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": meta["created_at"],
        "samples_total": len(samples),
        "samples_ok": n_success,
        "samples_error": n_failed,
        "metrics": {k: _aggregate(vs) for k, vs in metric_values.items()},
        "metrics_by_category": {
            cat: {k: _aggregate(vs) for k, vs in mv.items()} for cat, mv in metric_values_by_cat.items()
        },
    }
    _write_json(run_dir / "summary.json", summary)

    # Global run index for quick comparison across runs.
    index_path = out_root / "runs.csv"
    header = [
        "run_id",
        "created_at",
        "run_name",
        "manifest_path",
        "manifest_sha256",
        "workflow_sha256",
        "git_short",
        "git_dirty",
        "samples_total",
        "samples_ok",
        "samples_error",
        "psnr_mean",
        "ssim_mean",
        "psnr_masked_mean",
        "ssim_masked_mean",
        "pred_dir",
    ]
    metrics = summary.get("metrics", {})
    def mmean(key: str) -> str:
        v = metrics.get(key, {}).get("mean")
        return "" if v is None else f"{v:.6f}" if isinstance(v, float) else str(v)

    row = [
        run_id,
        meta["created_at"],
        _safe_slug(args.run_name) if args.run_name else "",
        str(manifest_path),
        manifest_hash,
        workflow_hash or "",
        git.get("short_commit") or "",
        str(git.get("dirty", False)),
        str(len(samples)),
        str(n_success),
        str(n_failed),
        mmean("psnr"),
        mmean("ssim"),
        mmean("psnr_masked"),
        mmean("ssim_masked"),
        str(pred_dir),
    ]
    _append_run_index_row(index_path, header, row)

    print(f"Run saved to: {run_dir}")
    print(f"Summary: {json_dumps(summary)}")


if __name__ == "__main__":
    main()
