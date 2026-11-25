#!/usr/bin/env python3
"""Run multiple deepfake-style forensic detectors on overlap CSV pairs.

The input should be a CSV built by `scripts/build_overlap_manifest.py`, which
contains two `pred_path_*` columns for fair pairwise comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import wilcoxon


EPS = 1e-8


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(
        np.float32
    )


@lru_cache(maxsize=512)
def _load_image(path_str: str) -> np.ndarray:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return rgb


def _safe_ratio(numer: float, denom: float) -> float:
    return float(numer / (denom + EPS))


def detector_fft_peakiness(rgb: np.ndarray) -> float:
    gray = _to_gray(rgb) - float(np.mean(_to_gray(rgb)))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    log_spec = np.log1p(spec)
    h, w = log_spec.shape
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask_hf = rr > (0.15 * min(h, w))
    vals = log_spec[mask_hf]
    if vals.size < 16:
        return math.nan
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + EPS
    p95 = float(np.percentile(vals, 95.0))
    return (p95 - med) / mad


def detector_jpeg_blockiness(rgb: np.ndarray) -> float:
    gray = _to_gray(rgb)
    h, w = gray.shape
    v_idx = np.arange(8, w, 8)
    h_idx = np.arange(8, h, 8)
    if v_idx.size == 0 or h_idx.size == 0:
        return math.nan

    vert_block = float(np.mean(np.abs(gray[:, v_idx] - gray[:, v_idx - 1])))
    horz_block = float(np.mean(np.abs(gray[h_idx, :] - gray[h_idx - 1, :])))
    vert_all = float(np.mean(np.abs(gray[:, 1:] - gray[:, :-1])))
    horz_all = float(np.mean(np.abs(gray[1:, :] - gray[:-1, :])))

    score_v = _safe_ratio(vert_block, vert_all)
    score_h = _safe_ratio(horz_block, horz_all)
    return 0.5 * (score_v + score_h)


def detector_noise_inconsistency(rgb: np.ndarray) -> float:
    gray = _to_gray(rgb)
    blur = ndimage.gaussian_filter(gray, sigma=1.2)
    residual = gray - blur
    local_var = ndimage.uniform_filter(residual * residual, size=9)
    mu = float(np.mean(local_var))
    sigma = float(np.std(local_var))
    return _safe_ratio(sigma, mu)


def detector_chroma_misalignment(rgb: np.ndarray) -> float:
    chans = [rgb[:, :, i] for i in range(3)]
    grads = []
    for ch in chans:
        gx = ndimage.sobel(ch, axis=1)
        gy = ndimage.sobel(ch, axis=0)
        grads.append(np.hypot(gx, gy))
    rg = np.mean(np.abs(grads[0] - grads[1]))
    rb = np.mean(np.abs(grads[0] - grads[2]))
    gb = np.mean(np.abs(grads[1] - grads[2]))
    return float((rg + rb + gb) / 3.0)


@dataclass(frozen=True)
class Detector:
    name: str
    score_fn: Callable[[np.ndarray], float]


DEFAULT_DETECTORS: Dict[str, Detector] = {
    "fft_peakiness": Detector("fft_peakiness", detector_fft_peakiness),
    "jpeg_blockiness": Detector("jpeg_blockiness", detector_jpeg_blockiness),
    "noise_inconsistency": Detector("noise_inconsistency", detector_noise_inconsistency),
    "chroma_misalignment": Detector("chroma_misalignment", detector_chroma_misalignment),
}


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 8:
        return None
    diff = a - b
    if np.allclose(diff, 0):
        return None
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
    except Exception:
        return None


def _pick_pred_cols(columns: Sequence[str], pred_a: str, pred_b: str) -> Tuple[str, str]:
    if pred_a and pred_b:
        if pred_a not in columns or pred_b not in columns:
            raise ValueError("Provided pred columns are not present in CSV.")
        return pred_a, pred_b
    pred_cols = [c for c in columns if c.startswith("pred_path_")]
    if len(pred_cols) < 2:
        raise ValueError("Need at least two `pred_path_*` columns in overlap CSV.")
    return pred_cols[0], pred_cols[1]


def _pick_detector_list(names_csv: str) -> List[Detector]:
    names = [x.strip() for x in names_csv.split(",") if x.strip()]
    if not names:
        names = list(DEFAULT_DETECTORS.keys())
    bad = [n for n in names if n not in DEFAULT_DETECTORS]
    if bad:
        raise ValueError(f"Unknown detector(s): {bad}. Known: {list(DEFAULT_DETECTORS)}")
    return [DEFAULT_DETECTORS[n] for n in names]

def _read_csv_columns(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k)) for k in fieldnames})


def _run(
    overlap_csv: Path,
    outdir: Path,
    pred_a_col: str,
    pred_b_col: str,
    person_col: str,
    detectors: Sequence[Detector],
    max_samples: int,
) -> None:
    run_a = pred_a_col.replace("pred_path_", "")
    run_b = pred_b_col.replace("pred_path_", "")

    rows: List[Dict[str, object]] = []

    with overlap_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in CSV: {overlap_csv}")

        for idx, row in enumerate(reader):
            if max_samples > 0 and idx >= max_samples:
                break

            pair_id = str(row.get("pair_id") or idx)
            path_a = str(row.get(pred_a_col) or "")
            path_b = str(row.get(pred_b_col) or "")
            person_path = str(row.get(person_col) or "") if person_col in reader.fieldnames else ""

            if not path_a or not path_b:
                continue

            try:
                img_a = _load_image(path_a)
                img_b = _load_image(path_b)
                img_person = _load_image(person_path) if person_path else None
            except Exception:
                continue

            for det in detectors:
                s_a = det.score_fn(img_a)
                s_b = det.score_fn(img_b)
                s_real = det.score_fn(img_person) if img_person is not None else math.nan
                rows.append(
                    {
                        "pair_id": pair_id,
                        "detector": det.name,
                        "run_a": run_a,
                        "run_b": run_b,
                        "score_a": float(s_a),
                        "score_b": float(s_b),
                        "score_real_person": float(s_real),
                        "delta_a_minus_b": float(s_a - s_b),
                    }
                )

    if not rows:
        raise RuntimeError("No valid samples were scored.")

    per_pair_rows: List[Dict[str, object]] = []
    for r in rows:
        s_a = float(r.get("score_a", math.nan))
        s_b = float(r.get("score_b", math.nan))
        if not (math.isfinite(s_a) and math.isfinite(s_b)):
            continue
        per_pair_rows.append(r)

    if not per_pair_rows:
        raise RuntimeError("No finite scores were produced.")

    summaries: List[Dict[str, object]] = []
    combined_z_rows: List[Dict[str, object]] = []

    per_detector: Dict[str, List[Dict[str, object]]] = {}
    for r in per_pair_rows:
        per_detector.setdefault(str(r["detector"]), []).append(r)

    for det in detectors:
        det_name = det.name
        det_rows = per_detector.get(det_name)
        if not det_rows:
            continue

        a = np.asarray([r["score_a"] for r in det_rows], dtype=np.float64)
        b = np.asarray([r["score_b"] for r in det_rows], dtype=np.float64)
        real_all = np.asarray([r.get("score_real_person", math.nan) for r in det_rows], dtype=np.float64)
        real = real_all[np.isfinite(real_all)]

        joined = np.concatenate([a, b])
        mu = float(np.mean(joined))
        sd = float(np.std(joined))
        if sd < EPS:
            sd = 1.0

        for r in det_rows:
            z_a = (float(r["score_a"]) - mu) / sd
            z_b = (float(r["score_b"]) - mu) / sd
            combined_z_rows.append(
                {
                    "pair_id": str(r["pair_id"]),
                    "detector": det_name,
                    "z_a": float(z_a),
                    "z_b": float(z_b),
                }
            )

        if real.size >= 8:
            threshold = float(np.mean(real) + 2.0 * np.std(real))
            threshold_source = "real_mean_plus_2std"
        else:
            threshold = float(np.percentile(joined, 75.0))
            threshold_source = "global_p75_fallback"

        summaries.append(
            {
                "detector": det_name,
                "n_pairs": int(a.size),
                "run_a_mean": float(np.mean(a)),
                "run_b_mean": float(np.mean(b)),
                "mean_delta_a_minus_b": float(np.mean(a - b)),
                "median_delta_a_minus_b": float(np.median(a - b)),
                "run_a_flag_rate": float(np.mean(a > threshold)),
                "run_b_flag_rate": float(np.mean(b > threshold)),
                "a_more_fake_win_rate": float(np.mean(a > b)),
                "paired_wilcoxon_p": _wilcoxon_p(a, b),
                "threshold": threshold,
                "threshold_source": threshold_source,
            }
        )

    if not combined_z_rows:
        raise RuntimeError("No detector z-scores were produced.")

    sum_za: Dict[str, float] = {}
    sum_zb: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in combined_z_rows:
        pid = str(r["pair_id"])
        sum_za[pid] = sum_za.get(pid, 0.0) + float(r["z_a"])
        sum_zb[pid] = sum_zb.get(pid, 0.0) + float(r["z_b"])
        counts[pid] = counts.get(pid, 0) + 1

    combined_rows: List[Dict[str, object]] = []
    for pid in sorted(counts.keys()):
        n = counts[pid]
        z_a = sum_za[pid] / n
        z_b = sum_zb[pid] / n
        combined_rows.append(
            {
                "pair_id": pid,
                "combined_z_a": float(z_a),
                "combined_z_b": float(z_b),
                "combined_delta_a_minus_b": float(z_a - z_b),
            }
        )

    a_comb = np.asarray([r["combined_z_a"] for r in combined_rows], dtype=np.float64)
    b_comb = np.asarray([r["combined_z_b"] for r in combined_rows], dtype=np.float64)
    combined_summary = {
        "detector": "combined_z_mean",
        "n_pairs": int(a_comb.size),
        "run_a_mean": float(np.mean(a_comb)),
        "run_b_mean": float(np.mean(b_comb)),
        "mean_delta_a_minus_b": float(np.mean(a_comb - b_comb)),
        "median_delta_a_minus_b": float(np.median(a_comb - b_comb)),
        "run_a_flag_rate": float(np.mean(a_comb > 0.5)),
        "run_b_flag_rate": float(np.mean(b_comb > 0.5)),
        "a_more_fake_win_rate": float(np.mean(a_comb > b_comb)),
        "paired_wilcoxon_p": _wilcoxon_p(a_comb, b_comb),
        "threshold": 0.5,
        "threshold_source": "zscore_fixed_0.5",
    }
    summaries.append(combined_summary)

    outdir.mkdir(parents=True, exist_ok=True)
    per_pair_path = outdir / "deepfake_per_pair.csv"
    combined_path = outdir / "deepfake_combined_per_pair.csv"
    summary_path = outdir / "deepfake_summary.csv"
    summary_json_path = outdir / "deepfake_summary.json"

    _write_csv(
        per_pair_path,
        per_pair_rows,
        fieldnames=[
            "pair_id",
            "detector",
            "run_a",
            "run_b",
            "score_a",
            "score_b",
            "score_real_person",
            "delta_a_minus_b",
        ],
    )
    _write_csv(
        combined_path,
        combined_rows,
        fieldnames=["pair_id", "combined_z_a", "combined_z_b", "combined_delta_a_minus_b"],
    )
    _write_csv(
        summary_path,
        summaries,
        fieldnames=[
            "detector",
            "n_pairs",
            "run_a_mean",
            "run_b_mean",
            "mean_delta_a_minus_b",
            "median_delta_a_minus_b",
            "run_a_flag_rate",
            "run_b_flag_rate",
            "a_more_fake_win_rate",
            "paired_wilcoxon_p",
            "threshold",
            "threshold_source",
        ],
    )
    summary_json_path.write_text(
        json.dumps(
            {
                "overlap_csv": str(overlap_csv),
                "pred_a_col": pred_a_col,
                "pred_b_col": pred_b_col,
                "run_a": run_a,
                "run_b": run_b,
                "detectors": [d.name for d in detectors],
                "rows_scored": int(len(per_pair_rows)),
                "pair_count": int(len(combined_rows)),
                "summary": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {per_pair_path}")
    print(f"Saved: {combined_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {summary_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--pred-a-col", default="")
    parser.add_argument("--pred-b-col", default="")
    parser.add_argument("--person-col", default="person_path")
    parser.add_argument(
        "--detectors",
        default="fft_peakiness,jpeg_blockiness,noise_inconsistency,chroma_misalignment",
        help="Comma-separated detector names.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns = _read_csv_columns(args.overlap_csv)
    pred_a_col, pred_b_col = _pick_pred_cols(columns, args.pred_a_col, args.pred_b_col)
    detectors = _pick_detector_list(args.detectors)
    _run(
        overlap_csv=args.overlap_csv,
        outdir=args.outdir,
        pred_a_col=pred_a_col,
        pred_b_col=pred_b_col,
        person_col=args.person_col,
        detectors=detectors,
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
