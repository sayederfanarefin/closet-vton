#!/usr/bin/env python3
"""Build a paper-ready report for non-stableviton person-cloth comparisons.

This script expects an eval root produced from comparisons anchored on:
  - ours_vs_hrviton
  - ours_vs_idm
  - ours_vs_catvton

It creates:
  <eval_root>/paper/report.md
  <eval_root>/paper/report.tex
  <eval_root>/paper/table_*.csv
  <eval_root>/paper/tables_*.tex
  <eval_root>/paper/figures/*.png
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - fallback when scipy missing
    wilcoxon = None


DETECTORS = [
    "fft_peakiness",
    "jpeg_blockiness",
    "noise_inconsistency",
    "chroma_misalignment",
]
FIDELITY_METRICS = ["cloth_score", "face_score", "pose_score"]
EPS = 1e-8


def _safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _infer_model(run_name: str) -> Optional[str]:
    n = (run_name or "").lower()
    if "stableviton" in n:
        return None
    if "mytest3curt" in n:
        return "Ours"
    if "hrviton" in n:
        return "HR-VITON"
    if "catvton" in n:
        return "CatVTON"
    if "idm" in n:
        return "IDM-VTON"
    return None


def _iter_latest_eval_root(runs_root: Path) -> Path:
    cands = sorted(p for p in runs_root.glob("eval_nonstable_personcloth_*") if p.is_dir())
    if not cands:
        raise FileNotFoundError(f"No eval_nonstable_personcloth_* under: {runs_root}")
    return cands[-1]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for row in reader:
            rows.append({k: (row.get(k) or "") for k in fields})
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            out: Dict[str, object] = {}
            for k in fieldnames:
                v = r.get(k)
                if isinstance(v, float) and not math.isfinite(v):
                    out[k] = ""
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = v
            w.writerow(out)


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if wilcoxon is None:
        return None
    if a.size < 8:
        return None
    diff = a - b
    if np.allclose(diff, 0.0):
        return None
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
    except Exception:
        return None


def _latex_escape(s: str) -> str:
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _fmt(x: object, digits: int = 4) -> str:
    if x is None:
        return "--"
    if isinstance(x, str):
        s = x.strip()
        return s if s else "--"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return "--"
    if abs(v) >= 1.0 and abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-4):
        return f"{v:.2e}"
    return f"{v:.{digits}f}"


def _fmt_pct(x: object, digits: int = 1) -> str:
    v = _safe_float(x)
    if v is None:
        return "--"
    return f"{100.0 * v:.{digits}f}%"


def _mean_std_threshold(real_vals: np.ndarray, joined_vals: np.ndarray) -> Tuple[float, str]:
    if real_vals.size >= 8:
        return float(np.mean(real_vals) + 2.0 * np.std(real_vals)), "real_mean_plus_2std"
    return float(np.percentile(joined_vals, 75.0)), "global_p75_fallback"


def _pair_ids_for_models(
    store: Dict[str, Dict[str, Dict[str, float]]], metric: str, a: str, b: str
) -> List[str]:
    out: List[str] = []
    for pid, by_model in store.get(metric, {}).items():
        if a in by_model and b in by_model:
            out.append(pid)
    out.sort()
    return out


def _arrays_for_pair(
    store: Dict[str, Dict[str, Dict[str, float]]], metric: str, pair_ids: Sequence[str], a: str, b: str
) -> Tuple[np.ndarray, np.ndarray]:
    va: List[float] = []
    vb: List[float] = []
    metric_store = store[metric]
    for pid in pair_ids:
        va.append(float(metric_store[pid][a]))
        vb.append(float(metric_store[pid][b]))
    return np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64)


def _pairwise_deepfake_rows(
    *,
    deep_scores: Dict[str, Dict[str, Dict[str, float]]],
    real_scores: Dict[str, Dict[str, float]],
    models: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    def combined_for_pair(a: str, b: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        pair_sets: List[set[str]] = []
        for det in DETECTORS:
            pair_sets.append(set(_pair_ids_for_models(deep_scores, det, a, b)))
        common = sorted(set.intersection(*pair_sets)) if pair_sets else []
        if not common:
            return [], np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

        z_a_all: List[np.ndarray] = []
        z_b_all: List[np.ndarray] = []
        for det in DETECTORS:
            arr_a, arr_b = _arrays_for_pair(deep_scores, det, common, a, b)
            joined = np.concatenate([arr_a, arr_b])
            mu = float(np.mean(joined))
            sd = float(np.std(joined))
            if sd < EPS:
                sd = 1.0
            z_a_all.append((arr_a - mu) / sd)
            z_b_all.append((arr_b - mu) / sd)
        z_a = np.mean(np.stack(z_a_all, axis=0), axis=0)
        z_b = np.mean(np.stack(z_b_all, axis=0), axis=0)
        return common, z_a, z_b

    for a, b in itertools.combinations(models, 2):
        for det in DETECTORS:
            pair_ids = _pair_ids_for_models(deep_scores, det, a, b)
            if not pair_ids:
                continue
            arr_a, arr_b = _arrays_for_pair(deep_scores, det, pair_ids, a, b)
            joined = np.concatenate([arr_a, arr_b])
            real = np.asarray(
                [_safe_float(real_scores.get(det, {}).get(pid)) for pid in pair_ids],
                dtype=np.float64,
            )
            real = real[np.isfinite(real)]
            threshold, threshold_source = _mean_std_threshold(real, joined)

            tol = 1e-12
            tie = np.abs(arr_a - arr_b) <= tol
            win = arr_a < (arr_b - tol)  # lower is better

            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "detector": det,
                    "n_pairs": int(arr_a.size),
                    "mean_a": float(np.mean(arr_a)),
                    "mean_b": float(np.mean(arr_b)),
                    "mean_delta_a_minus_b": float(np.mean(arr_a - arr_b)),
                    "median_delta_a_minus_b": float(np.median(arr_a - arr_b)),
                    "a_better_win_rate": float(np.mean(win)),
                    "tie_rate": float(np.mean(tie)),
                    "flag_rate_a": float(np.mean(arr_a > threshold)),
                    "flag_rate_b": float(np.mean(arr_b > threshold)),
                    "threshold": float(threshold),
                    "threshold_source": threshold_source,
                    "paired_wilcoxon_p": _wilcoxon_p(arr_a, arr_b),
                }
            )

        pair_ids, z_a, z_b = combined_for_pair(a, b)
        if pair_ids:
            tol = 1e-12
            tie = np.abs(z_a - z_b) <= tol
            win = z_a < (z_b - tol)
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "detector": "combined_z_mean",
                    "n_pairs": int(z_a.size),
                    "mean_a": float(np.mean(z_a)),
                    "mean_b": float(np.mean(z_b)),
                    "mean_delta_a_minus_b": float(np.mean(z_a - z_b)),
                    "median_delta_a_minus_b": float(np.median(z_a - z_b)),
                    "a_better_win_rate": float(np.mean(win)),
                    "tie_rate": float(np.mean(tie)),
                    "flag_rate_a": float(np.mean(z_a > 0.5)),
                    "flag_rate_b": float(np.mean(z_b > 0.5)),
                    "threshold": 0.5,
                    "threshold_source": "zscore_fixed_0.5",
                    "paired_wilcoxon_p": _wilcoxon_p(z_a, z_b),
                }
            )

    rows.sort(key=lambda r: (str(r["detector"]), str(r["model_a"]), str(r["model_b"])))
    return rows


def _pairwise_fidelity_rows(
    *,
    fid_scores: Dict[str, Dict[str, Dict[str, float]]],
    models: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for a, b in itertools.combinations(models, 2):
        for metric in FIDELITY_METRICS:
            pair_ids = _pair_ids_for_models(fid_scores, metric, a, b)
            if not pair_ids:
                continue
            arr_a, arr_b = _arrays_for_pair(fid_scores, metric, pair_ids, a, b)
            tol = 1e-12
            tie = np.abs(arr_a - arr_b) <= tol
            win = arr_a > (arr_b + tol)  # higher is better
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "n_pairs": int(arr_a.size),
                    "mean_a": float(np.mean(arr_a)),
                    "mean_b": float(np.mean(arr_b)),
                    "mean_delta_a_minus_b": float(np.mean(arr_a - arr_b)),
                    "median_delta_a_minus_b": float(np.median(arr_a - arr_b)),
                    "a_better_win_rate": float(np.mean(win)),
                    "tie_rate": float(np.mean(tie)),
                    "paired_wilcoxon_p": _wilcoxon_p(arr_a, arr_b),
                }
            )

        pair_sets = [set(_pair_ids_for_models(fid_scores, m, a, b)) for m in FIDELITY_METRICS]
        common = sorted(set.intersection(*pair_sets)) if pair_sets else []
        if common:
            comp_a_parts: List[np.ndarray] = []
            comp_b_parts: List[np.ndarray] = []
            for metric in FIDELITY_METRICS:
                arr_a, arr_b = _arrays_for_pair(fid_scores, metric, common, a, b)
                comp_a_parts.append(arr_a)
                comp_b_parts.append(arr_b)
            comp_a = np.mean(np.stack(comp_a_parts, axis=0), axis=0)
            comp_b = np.mean(np.stack(comp_b_parts, axis=0), axis=0)
            tol = 1e-12
            tie = np.abs(comp_a - comp_b) <= tol
            win = comp_a > (comp_b + tol)
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "metric": "fidelity_composite",
                    "n_pairs": int(comp_a.size),
                    "mean_a": float(np.mean(comp_a)),
                    "mean_b": float(np.mean(comp_b)),
                    "mean_delta_a_minus_b": float(np.mean(comp_a - comp_b)),
                    "median_delta_a_minus_b": float(np.median(comp_a - comp_b)),
                    "a_better_win_rate": float(np.mean(win)),
                    "tie_rate": float(np.mean(tie)),
                    "paired_wilcoxon_p": _wilcoxon_p(comp_a, comp_b),
                }
            )

    rows.sort(key=lambda r: (str(r["metric"]), str(r["model_a"]), str(r["model_b"])))
    return rows


def _model_summary_rows(
    *,
    deep_scores: Dict[str, Dict[str, Dict[str, float]]],
    real_scores: Dict[str, Dict[str, float]],
    fid_scores: Dict[str, Dict[str, Dict[str, float]]],
    models: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    detector_pair_sets: Dict[str, List[set[str]]] = {}
    for det in DETECTORS:
        detector_pair_sets[det] = []
        for m in models:
            s = set(pid for pid, by_m in deep_scores.get(det, {}).items() if m in by_m)
            detector_pair_sets[det].append(s)

    fid_pair_sets: Dict[str, List[set[str]]] = {}
    for metric in FIDELITY_METRICS:
        fid_pair_sets[metric] = []
        for m in models:
            s = set(pid for pid, by_m in fid_scores.get(metric, {}).items() if m in by_m)
            fid_pair_sets[metric].append(s)

    # Global common pairs for combined z across all models + detectors.
    deep_global_sets: List[set[str]] = []
    for det in DETECTORS:
        deep_global_sets.append(set.intersection(*detector_pair_sets[det]))
    deep_global_pairs = sorted(set.intersection(*deep_global_sets)) if deep_global_sets else []

    z_by_model: Dict[str, np.ndarray] = {}
    if deep_global_pairs:
        z_parts: Dict[str, List[np.ndarray]] = {m: [] for m in models}
        for det in DETECTORS:
            vals_per_model: Dict[str, np.ndarray] = {}
            for m in models:
                vals_per_model[m] = np.asarray(
                    [deep_scores[det][pid][m] for pid in deep_global_pairs],
                    dtype=np.float64,
                )
            joined = np.concatenate([vals_per_model[m] for m in models])
            mu = float(np.mean(joined))
            sd = float(np.std(joined))
            if sd < EPS:
                sd = 1.0
            for m in models:
                z_parts[m].append((vals_per_model[m] - mu) / sd)
        for m in models:
            z_by_model[m] = np.mean(np.stack(z_parts[m], axis=0), axis=0)

    # Global common pairs for fidelity composite.
    fid_global_sets: List[set[str]] = []
    for metric in FIDELITY_METRICS:
        fid_global_sets.append(set.intersection(*fid_pair_sets[metric]))
    fid_global_pairs = sorted(set.intersection(*fid_global_sets)) if fid_global_sets else []

    comp_by_model: Dict[str, np.ndarray] = {}
    if fid_global_pairs:
        for m in models:
            parts: List[np.ndarray] = []
            for metric in FIDELITY_METRICS:
                parts.append(
                    np.asarray([fid_scores[metric][pid][m] for pid in fid_global_pairs], dtype=np.float64)
                )
            comp_by_model[m] = np.mean(np.stack(parts, axis=0), axis=0)

    for m in models:
        row: Dict[str, object] = {"model": m}
        for det in DETECTORS:
            pair_ids = sorted(set(pid for pid, by_m in deep_scores.get(det, {}).items() if m in by_m))
            arr = np.asarray([deep_scores[det][pid][m] for pid in pair_ids], dtype=np.float64) if pair_ids else np.asarray([])
            row[f"{det}_mean"] = float(np.mean(arr)) if arr.size else None
            real = np.asarray(
                [_safe_float(real_scores.get(det, {}).get(pid)) for pid in pair_ids],
                dtype=np.float64,
            )
            real = real[np.isfinite(real)]
            joined = arr if arr.size else np.asarray([0.0], dtype=np.float64)
            threshold, _ = _mean_std_threshold(real, joined)
            row[f"{det}_flag_rate"] = float(np.mean(arr > threshold)) if arr.size else None

        if m in z_by_model:
            z = z_by_model[m]
            row["combined_z_mean"] = float(np.mean(z))
            row["combined_z_flag_rate"] = float(np.mean(z > 0.5))
            row["n_deep_common"] = int(z.size)
        else:
            row["combined_z_mean"] = None
            row["combined_z_flag_rate"] = None
            row["n_deep_common"] = 0

        for metric in FIDELITY_METRICS:
            pair_ids = sorted(set(pid for pid, by_m in fid_scores.get(metric, {}).items() if m in by_m))
            arr = np.asarray([fid_scores[metric][pid][m] for pid in pair_ids], dtype=np.float64) if pair_ids else np.asarray([])
            row[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else None

        if m in comp_by_model:
            row["fidelity_composite_mean"] = float(np.mean(comp_by_model[m]))
            row["n_fid_common"] = int(comp_by_model[m].size)
        else:
            row["fidelity_composite_mean"] = None
            row["n_fid_common"] = 0

        rows.append(row)

    rows.sort(key=lambda r: str(r["model"]))
    return rows


def _plot_grouped_bars(
    *,
    models: Sequence[str],
    categories: Sequence[str],
    values: Dict[str, Dict[str, float]],
    out_path: Path,
    title: str,
    y_label: str,
) -> None:
    x = np.arange(len(categories), dtype=np.float64)
    width = 0.75 / max(1, len(models))
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.5), constrained_layout=True)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, m in enumerate(models):
        vals = [float(values.get(m, {}).get(c, math.nan)) for c in categories]
        offs = (i - (len(models) - 1) / 2.0) * width
        ax.bar(x + offs, vals, width=width, label=m, color=palette[i % len(palette)], alpha=0.9)

    ax.set_xticks(x, categories)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(4, len(models)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_heatmap(
    *,
    models: Sequence[str],
    matrix: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str = "RdBu_r",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    if vmax < 1e-9:
        vmax = 1.0
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(models)), models, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(models)), models)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row advantage", rotation=90)
    for i in range(len(models)):
        for j in range(len(models)):
            v = matrix[i, j]
            txt = "--" if not math.isfinite(float(v)) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _find_font(size: int) -> ImageFont.ImageFont:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fit_tile(path: str, tile_w: int, tile_h: int) -> Image.Image:
    try:
        with Image.open(path) as im:
            src = im.convert("RGB")
    except Exception:
        src = Image.new("RGB", (tile_w, tile_h), color=(230, 230, 230))
    ratio_src = src.width / max(1, src.height)
    ratio_dst = tile_w / max(1, tile_h)
    if ratio_src >= ratio_dst:
        new_w = tile_w
        new_h = max(1, int(round(tile_w / ratio_src)))
    else:
        new_h = tile_h
        new_w = max(1, int(round(tile_h * ratio_src)))
    src = src.resize((new_w, new_h), Image.BILINEAR)
    tile = Image.new("RGB", (tile_w, tile_h), color=(0, 0, 0))
    tile.paste(src, ((tile_w - new_w) // 2, (tile_h - new_h) // 2))
    return tile


def _build_qualitative_grid(
    *,
    out_path: Path,
    pair_ids: Sequence[str],
    model_order: Sequence[str],
    pred_paths: Dict[str, Dict[str, str]],
    person_paths: Dict[str, str],
    cloth_paths: Dict[str, str],
) -> None:
    if not pair_ids:
        return
    cols = ["person", "cloth", *model_order]
    tile_w, tile_h = 240, 300
    pad = 12
    header_h = 34
    row_h = tile_h + 24
    width = pad + len(cols) * (tile_w + pad)
    height = pad + header_h + len(pair_ids) * (row_h + pad)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_h = _find_font(16)
    font_s = _find_font(12)

    for c, col in enumerate(cols):
        x = pad + c * (tile_w + pad)
        draw.text((x, pad), col, fill=(0, 0, 0), font=font_h)

    y0 = pad + header_h
    for r, pid in enumerate(pair_ids):
        y = y0 + r * (row_h + pad)
        draw.text((pad, y), pid, fill=(0, 0, 0), font=font_s)
        imgs: List[Tuple[str, str]] = [
            ("person", person_paths.get(pid, "")),
            ("cloth", cloth_paths.get(pid, "")),
        ]
        for m in model_order:
            imgs.append((m, pred_paths.get(m, {}).get(pid, "")))

        for c, (_, p) in enumerate(imgs):
            x = pad + c * (tile_w + pad)
            tile = _fit_tile(p, tile_w, tile_h)
            canvas.paste(tile, (x, y + 18))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _md_table(rows: Sequence[Dict[str, object]], cols: Sequence[str], pretty: Dict[str, str]) -> List[str]:
    out: List[str] = []
    out.append("| " + " | ".join(pretty.get(c, c) for c in cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if c in {"n_pairs", "n_deep_common", "n_fid_common"} or c.startswith("n_"):
                vals.append(_fmt(v, digits=0))
            elif "rate" in c or c.startswith("flag_") or c.endswith("_flag"):
                vals.append(_fmt_pct(v, digits=1))
            else:
                vals.append(_fmt(v, digits=4))
        out.append("| " + " | ".join(vals) + " |")
    return out


def _write_latex_table(
    *,
    path: Path,
    caption: str,
    label: str,
    rows: Sequence[Dict[str, object]],
    cols: Sequence[str],
    pretty: Dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * min(2, len(cols)) + "r" * max(0, len(cols) - 2)
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_latex_escape(pretty.get(c, c)) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        vals: List[str] = []
        for c in cols:
            v = r.get(c)
            if c in {"n_pairs", "n_deep_common", "n_fid_common"} or c.startswith("n_"):
                vals.append(_latex_escape(_fmt(v, digits=0)))
            elif "rate" in c or c.startswith("flag_") or c.endswith("_flag"):
                vals.append(_latex_escape(_fmt_pct(v, digits=1)))
            else:
                vals.append(_latex_escape(_fmt(v, digits=4)))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    # Labels should remain plain LaTeX identifiers (no escaping).
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--eval-root", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None, help="Defaults to <eval_root>/paper")
    p.add_argument("--top-k-qualitative", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    eval_root = args.eval_root.resolve() if args.eval_root else _iter_latest_eval_root(runs_root)
    out_dir = (args.outdir.resolve() if args.outdir else (eval_root / "paper").resolve())
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    deep_scores: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    real_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    fid_scores: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    pred_paths: Dict[str, Dict[str, str]] = defaultdict(dict)
    person_paths: Dict[str, str] = {}
    cloth_paths: Dict[str, str] = {}
    models_seen: set[str] = set()

    # Load anchor comparisons (ours_vs_*).
    comp_dirs = sorted(p for p in eval_root.iterdir() if p.is_dir() and p.name.startswith("ours_vs_"))
    if not comp_dirs:
        raise FileNotFoundError(f"No ours_vs_* directories found under: {eval_root}")

    for comp in comp_dirs:
        deep_path = comp / "deepfake" / "deepfake_per_pair.csv"
        fid_path = comp / "fidelity" / "fidelity_per_pair.csv"
        if not deep_path.exists() or not fid_path.exists():
            continue

        for row in _read_csv_rows(deep_path):
            det = row.get("detector", "")
            pid = row.get("pair_id", "")
            if not det or not pid:
                continue
            ma = _infer_model(row.get("run_a", ""))
            mb = _infer_model(row.get("run_b", ""))
            sa = _safe_float(row.get("score_a"))
            sb = _safe_float(row.get("score_b"))
            sr = _safe_float(row.get("score_real_person"))
            deep_scores.setdefault(det, {}).setdefault(pid, {})
            if ma and sa is not None:
                deep_scores[det][pid][ma] = float(sa)
                models_seen.add(ma)
            if mb and sb is not None:
                deep_scores[det][pid][mb] = float(sb)
                models_seen.add(mb)
            if sr is not None:
                real_scores.setdefault(det, {})[pid] = float(sr)

        for row in _read_csv_rows(fid_path):
            pid = row.get("pair_id", "")
            if not pid:
                continue
            ma = _infer_model(row.get("run_a", ""))
            mb = _infer_model(row.get("run_b", ""))
            for metric in FIDELITY_METRICS:
                sa = _safe_float(row.get(f"{metric}_a"))
                sb = _safe_float(row.get(f"{metric}_b"))
                fid_scores.setdefault(metric, {}).setdefault(pid, {})
                if ma and sa is not None:
                    fid_scores[metric][pid][ma] = float(sa)
                    models_seen.add(ma)
                if mb and sb is not None:
                    fid_scores[metric][pid][mb] = float(sb)
                    models_seen.add(mb)

    # Load overlap CSVs for qualitative image paths.
    overlap_paths = sorted(eval_root.glob("overlap_ours_vs_*.csv"))
    for overlap in overlap_paths:
        rows = _read_csv_rows(overlap)
        if not rows:
            continue
        fields = list(rows[0].keys())
        pred_cols = [c for c in fields if c.startswith("pred_path_")]
        cloth_cols = [c for c in fields if c.startswith("cloth_path_")]
        run_to_model: Dict[str, str] = {}
        for c in pred_cols:
            run_name = c.replace("pred_path_", "")
            model = _infer_model(run_name)
            if model:
                run_to_model[run_name] = model
        for row in rows:
            pid = row.get("pair_id", "")
            if not pid:
                continue
            if row.get("person_path"):
                person_paths[pid] = row["person_path"]
            # Prefer cloth path from ours if present.
            ours_cloth = None
            for cc in cloth_cols:
                rname = cc.replace("cloth_path_", "")
                if _infer_model(rname) == "Ours":
                    ours_cloth = row.get(cc, "")
                    break
            if ours_cloth:
                cloth_paths[pid] = ours_cloth
            elif cloth_cols and row.get(cloth_cols[0], ""):
                cloth_paths[pid] = row[cloth_cols[0]]

            for pc in pred_cols:
                rname = pc.replace("pred_path_", "")
                model = run_to_model.get(rname)
                if model and row.get(pc, ""):
                    pred_paths[model][pid] = row[pc]

    preferred_order = ["Ours", "HR-VITON", "IDM-VTON", "CatVTON"]
    models = [m for m in preferred_order if m in models_seen]
    if len(models) < 2:
        raise RuntimeError("Not enough models discovered to build pairwise report.")

    deep_rows = _pairwise_deepfake_rows(deep_scores=deep_scores, real_scores=real_scores, models=models)
    fid_rows = _pairwise_fidelity_rows(fid_scores=fid_scores, models=models)
    summary_rows = _model_summary_rows(
        deep_scores=deep_scores, real_scores=real_scores, fid_scores=fid_scores, models=models
    )

    deep_fields = [
        "model_a",
        "model_b",
        "detector",
        "n_pairs",
        "mean_a",
        "mean_b",
        "mean_delta_a_minus_b",
        "median_delta_a_minus_b",
        "a_better_win_rate",
        "tie_rate",
        "flag_rate_a",
        "flag_rate_b",
        "threshold",
        "threshold_source",
        "paired_wilcoxon_p",
    ]
    fid_fields = [
        "model_a",
        "model_b",
        "metric",
        "n_pairs",
        "mean_a",
        "mean_b",
        "mean_delta_a_minus_b",
        "median_delta_a_minus_b",
        "a_better_win_rate",
        "tie_rate",
        "paired_wilcoxon_p",
    ]
    summary_fields = [
        "model",
        "n_deep_common",
        "fft_peakiness_mean",
        "jpeg_blockiness_mean",
        "noise_inconsistency_mean",
        "chroma_misalignment_mean",
        "combined_z_mean",
        "combined_z_flag_rate",
        "n_fid_common",
        "cloth_score_mean",
        "face_score_mean",
        "pose_score_mean",
        "fidelity_composite_mean",
    ]

    _write_csv(out_dir / "table_pairwise_deepfake.csv", deep_rows, deep_fields)
    _write_csv(out_dir / "table_pairwise_fidelity.csv", fid_rows, fid_fields)
    _write_csv(out_dir / "table_model_summary.csv", summary_rows, summary_fields)

    # Figures.
    fid_values: Dict[str, Dict[str, float]] = {}
    for r in summary_rows:
        m = str(r["model"])
        fid_values[m] = {
            "cloth_score": float(r["cloth_score_mean"]) if r["cloth_score_mean"] is not None else math.nan,
            "face_score": float(r["face_score_mean"]) if r["face_score_mean"] is not None else math.nan,
            "pose_score": float(r["pose_score_mean"]) if r["pose_score_mean"] is not None else math.nan,
            "fidelity_composite": float(r["fidelity_composite_mean"])
            if r["fidelity_composite_mean"] is not None
            else math.nan,
        }
    _plot_grouped_bars(
        models=models,
        categories=["cloth_score", "face_score", "pose_score", "fidelity_composite"],
        values=fid_values,
        out_path=fig_dir / "fig_fidelity_means.png",
        title="Fidelity Means by Model (higher is better)",
        y_label="score",
    )

    deep_values: Dict[str, Dict[str, float]] = {}
    for r in summary_rows:
        m = str(r["model"])
        deep_values[m] = {
            "fft_peakiness": float(r["fft_peakiness_mean"]) if r["fft_peakiness_mean"] is not None else math.nan,
            "jpeg_blockiness": float(r["jpeg_blockiness_mean"]) if r["jpeg_blockiness_mean"] is not None else math.nan,
            "noise_inconsistency": float(r["noise_inconsistency_mean"])
            if r["noise_inconsistency_mean"] is not None
            else math.nan,
            "chroma_misalignment": float(r["chroma_misalignment_mean"])
            if r["chroma_misalignment_mean"] is not None
            else math.nan,
        }
    _plot_grouped_bars(
        models=models,
        categories=DETECTORS,
        values=deep_values,
        out_path=fig_dir / "fig_deepfake_means.png",
        title="Deepfake Detector Means by Model (lower is better)",
        y_label="detector score",
    )

    # Build heatmaps from composite rows.
    idx = {m: i for i, m in enumerate(models)}
    mat_fid = np.zeros((len(models), len(models)), dtype=np.float64)
    mat_deep = np.zeros((len(models), len(models)), dtype=np.float64)
    mat_fid[:] = np.nan
    mat_deep[:] = np.nan
    for i in range(len(models)):
        mat_fid[i, i] = 0.0
        mat_deep[i, i] = 0.0

    for r in fid_rows:
        if r.get("metric") != "fidelity_composite":
            continue
        a = str(r["model_a"])
        b = str(r["model_b"])
        d = _safe_float(r.get("mean_delta_a_minus_b"))
        if d is None:
            continue
        ia, ib = idx[a], idx[b]
        mat_fid[ia, ib] = float(d)  # higher is better
        mat_fid[ib, ia] = -float(d)

    for r in deep_rows:
        if r.get("detector") != "combined_z_mean":
            continue
        a = str(r["model_a"])
        b = str(r["model_b"])
        d = _safe_float(r.get("mean_delta_a_minus_b"))
        if d is None:
            continue
        ia, ib = idx[a], idx[b]
        # Deepfake is lower-is-better, so row advantage = mean_b - mean_a = -delta.
        mat_deep[ia, ib] = -float(d)
        mat_deep[ib, ia] = float(d)

    _plot_heatmap(
        models=models,
        matrix=mat_fid,
        out_path=fig_dir / "fig_pairwise_fidelity_heatmap.png",
        title="Pairwise Fidelity Composite Advantage (row - col, higher is better)",
    )
    _plot_heatmap(
        models=models,
        matrix=mat_deep,
        out_path=fig_dir / "fig_pairwise_deepfake_heatmap.png",
        title="Pairwise Deepfake Combined Advantage (row - col, higher is better)",
    )

    # Qualitative grid selection.
    if "Ours" in models and len(models) >= 3:
        common_pairs = sorted(
            set(person_paths.keys())
            & set(cloth_paths.keys())
            & set.intersection(*(set(pred_paths.get(m, {}).keys()) for m in models))
        )
        if common_pairs:
            # Build per-pair fidelity composite for each model (using available metrics).
            fid_pair_comp: Dict[str, Dict[str, float]] = {m: {} for m in models}
            for pid in common_pairs:
                for m in models:
                    vals = []
                    for met in FIDELITY_METRICS:
                        v = fid_scores.get(met, {}).get(pid, {}).get(m)
                        if v is not None:
                            vals.append(float(v))
                    if vals:
                        fid_pair_comp[m][pid] = float(np.mean(np.asarray(vals, dtype=np.float64)))

            # Build global combined z per model/pair.
            z_pair: Dict[str, Dict[str, float]] = {m: {} for m in models}
            # Use detectors where all models have the pair.
            det_pair_intersection = []
            for det in DETECTORS:
                det_pair_intersection.append(
                    set(pid for pid in common_pairs if all(m in deep_scores.get(det, {}).get(pid, {}) for m in models))
                )
            z_pairs = sorted(set.intersection(*det_pair_intersection)) if det_pair_intersection else []
            if z_pairs:
                z_parts: Dict[str, List[np.ndarray]] = {m: [] for m in models}
                for det in DETECTORS:
                    vals = {m: np.asarray([deep_scores[det][pid][m] for pid in z_pairs], dtype=np.float64) for m in models}
                    joined = np.concatenate([vals[m] for m in models])
                    mu = float(np.mean(joined))
                    sd = float(np.std(joined))
                    if sd < EPS:
                        sd = 1.0
                    for m in models:
                        z_parts[m].append((vals[m] - mu) / sd)
                for m in models:
                    z_m = np.mean(np.stack(z_parts[m], axis=0), axis=0)
                    for i, pid in enumerate(z_pairs):
                        z_pair[m][pid] = float(z_m[i])

            baselines = [m for m in models if m != "Ours"]
            ranked: List[Tuple[float, str]] = []
            for pid in common_pairs:
                if pid not in fid_pair_comp["Ours"]:
                    continue
                if pid not in z_pair["Ours"]:
                    continue
                if any(pid not in fid_pair_comp[b] for b in baselines):
                    continue
                if any(pid not in z_pair[b] for b in baselines):
                    continue
                fid_adv = fid_pair_comp["Ours"][pid] - float(np.mean([fid_pair_comp[b][pid] for b in baselines]))
                deep_adv = float(np.mean([z_pair[b][pid] for b in baselines])) - z_pair["Ours"][pid]
                ranked.append((fid_adv + deep_adv, pid))
            ranked.sort(reverse=True, key=lambda t: t[0])
            top_pairs = [pid for _, pid in ranked[: max(1, int(args.top_k_qualitative))]]
            _build_qualitative_grid(
                out_path=fig_dir / "fig_qualitative_top_pairs.png",
                pair_ids=top_pairs,
                model_order=models,
                pred_paths=pred_paths,
                person_paths=person_paths,
                cloth_paths=cloth_paths,
            )

    # Markdown report.
    pretty_deep = {
        "model_a": "Model A",
        "model_b": "Model B",
        "detector": "Detector",
        "n_pairs": "N",
        "mean_a": "A mean",
        "mean_b": "B mean",
        "mean_delta_a_minus_b": "mean(A-B)",
        "median_delta_a_minus_b": "median(A-B)",
        "a_better_win_rate": "A better win",
        "tie_rate": "Tie",
        "flag_rate_a": "A flag",
        "flag_rate_b": "B flag",
        "threshold": "Threshold",
        "threshold_source": "Threshold src",
        "paired_wilcoxon_p": "p",
    }
    pretty_fid = {
        "model_a": "Model A",
        "model_b": "Model B",
        "metric": "Metric",
        "n_pairs": "N",
        "mean_a": "A mean",
        "mean_b": "B mean",
        "mean_delta_a_minus_b": "mean(A-B)",
        "median_delta_a_minus_b": "median(A-B)",
        "a_better_win_rate": "A better win",
        "tie_rate": "Tie",
        "paired_wilcoxon_p": "p",
    }
    pretty_sum = {
        "model": "Model",
        "n_deep_common": "N deep",
        "fft_peakiness_mean": "fft",
        "jpeg_blockiness_mean": "jpeg",
        "noise_inconsistency_mean": "noise",
        "chroma_misalignment_mean": "chroma",
        "combined_z_mean": "combined z",
        "combined_z_flag_rate": "combined flag",
        "n_fid_common": "N fid",
        "cloth_score_mean": "cloth",
        "face_score_mean": "face",
        "pose_score_mean": "pose",
        "fidelity_composite_mean": "fidelity comp",
    }

    md: List[str] = []
    md.append("# Non-StableVITON Comparative Evaluation (Paper Ready)")
    md.append("")
    md.append(f"- eval_root: `{eval_root}`")
    md.append(f"- generated: `{out_dir}`")
    md.append("- scope: `Ours`, `HR-VITON`, `IDM-VTON`, `CatVTON` (StableVITON excluded)")
    md.append("- sample policy: first `N=1000` overlap pairs per anchor comparison")
    md.append("")
    md.append("Notes:")
    md.append("- Deepfake metrics: lower is better.")
    md.append("- Fidelity metrics: higher is better.")
    md.append("- Pairwise tables include all model-pair combinations.")
    md.append("")
    md.append("## Model Summary")
    md.append("")
    md.extend(_md_table(summary_rows, summary_fields, pretty_sum))
    md.append("")
    md.append("## Figures")
    md.append("")
    md.append("![](figures/fig_fidelity_means.png)")
    md.append("![](figures/fig_deepfake_means.png)")
    md.append("![](figures/fig_pairwise_fidelity_heatmap.png)")
    md.append("![](figures/fig_pairwise_deepfake_heatmap.png)")
    if (fig_dir / "fig_qualitative_top_pairs.png").exists():
        md.append("![](figures/fig_qualitative_top_pairs.png)")
    md.append("")
    md.append("## Pairwise Deepfake (All Comparisons)")
    md.append("")
    md.extend(_md_table(deep_rows, deep_fields, pretty_deep))
    md.append("")
    md.append("## Pairwise Fidelity (All Comparisons)")
    md.append("")
    md.extend(_md_table(fid_rows, fid_fields, pretty_fid))
    md.append("")
    md.append("## Anchor Comparison Visual Reports")
    md.append("")
    md.append(f"- Main viz report: `{eval_root / 'viz' / 'report.md'}`")
    md.append("- Per-anchor plots are available under `viz/ours_vs_*`.")
    md.append("")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    # LaTeX tables.
    _write_latex_table(
        path=out_dir / "tables_model_summary.tex",
        caption="Model-level summary on non-StableVITON comparison set.",
        label="tab:model_summary_nonstable",
        rows=summary_rows,
        cols=summary_fields,
        pretty=pretty_sum,
    )
    _write_latex_table(
        path=out_dir / "tables_pairwise_deepfake.tex",
        caption="All pairwise deepfake detector comparisons (lower is better).",
        label="tab:pairwise_deepfake_nonstable",
        rows=deep_rows,
        cols=deep_fields,
        pretty=pretty_deep,
    )
    _write_latex_table(
        path=out_dir / "tables_pairwise_fidelity.tex",
        caption="All pairwise fidelity comparisons (higher is better).",
        label="tab:pairwise_fidelity_nonstable",
        rows=fid_rows,
        cols=fid_fields,
        pretty=pretty_fid,
    )

    tex_lines: List[str] = []
    tex_lines.append("\\documentclass{article}")
    tex_lines.append("\\usepackage[margin=1in]{geometry}")
    tex_lines.append("\\usepackage{graphicx}")
    tex_lines.append("\\usepackage{booktabs}")
    tex_lines.append("\\usepackage{float}")
    tex_lines.append("\\begin{document}")
    tex_lines.append("\\section*{Non-StableVITON Comparative Evaluation}")
    tex_lines.append(
        "Scope: Ours, HR-VITON, IDM-VTON, CatVTON (StableVITON excluded). "
        "All numbers are computed on first $N=1000$ overlap pairs per anchor comparison."
    )
    tex_lines.append("\\input{tables_model_summary.tex}")
    tex_lines.append("\\begin{figure}[H]\\centering")
    tex_lines.append("\\includegraphics[width=0.95\\textwidth]{figures/fig_fidelity_means.png}")
    tex_lines.append("\\caption{Fidelity means by model (higher is better).}")
    tex_lines.append("\\end{figure}")
    tex_lines.append("\\begin{figure}[H]\\centering")
    tex_lines.append("\\includegraphics[width=0.95\\textwidth]{figures/fig_deepfake_means.png}")
    tex_lines.append("\\caption{Deepfake detector means by model (lower is better).}")
    tex_lines.append("\\end{figure}")
    tex_lines.append("\\begin{figure}[H]\\centering")
    tex_lines.append("\\includegraphics[width=0.9\\textwidth]{figures/fig_pairwise_fidelity_heatmap.png}")
    tex_lines.append("\\caption{Pairwise fidelity composite advantage matrix (row - col).}")
    tex_lines.append("\\end{figure}")
    tex_lines.append("\\begin{figure}[H]\\centering")
    tex_lines.append("\\includegraphics[width=0.9\\textwidth]{figures/fig_pairwise_deepfake_heatmap.png}")
    tex_lines.append("\\caption{Pairwise deepfake combined advantage matrix (row - col).}")
    tex_lines.append("\\end{figure}")
    if (fig_dir / "fig_qualitative_top_pairs.png").exists():
        tex_lines.append("\\begin{figure}[H]\\centering")
        tex_lines.append("\\includegraphics[width=\\textwidth]{figures/fig_qualitative_top_pairs.png}")
        tex_lines.append("\\caption{Qualitative top pairs where Ours shows strongest combined advantage.}")
        tex_lines.append("\\end{figure}")
    tex_lines.append("\\input{tables_pairwise_deepfake.tex}")
    tex_lines.append("\\input{tables_pairwise_fidelity.tex}")
    tex_lines.append("\\end{document}")
    (out_dir / "report.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    print(f"Saved markdown: {out_dir / 'report.md'}")
    print(f"Saved latex:    {out_dir / 'report.tex'}")
    print(f"Saved figures:  {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
