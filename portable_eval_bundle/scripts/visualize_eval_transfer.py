#!/usr/bin/env python3
"""Visualize eval_transfer outputs (deepfake + fidelity) as plots and image grids.

This script reads the outputs produced by:
  bash portable_eval_bundle/run_all_comparisons.sh .

It produces:
  <eval_root>/viz/<comparison_name>/*.png
  <eval_root>/viz/report.md

The plotting is intentionally implemented with Pillow (no matplotlib/pandas),
so it works in minimal environments where compiled wheels may be incompatible
with NumPy 2.x.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RGB = Tuple[int, int, int]

DETECTOR_DESCRIPTIONS: Dict[str, str] = {
    "fft_peakiness": "Peakiness of high-frequency FFT magnitude (captures periodic/high-frequency artifacts).",
    "jpeg_blockiness": "Strength of 8x8 block boundary discontinuities (JPEG-like blocking).",
    "noise_inconsistency": "Spatial inconsistency of local noise residual variance.",
    "chroma_misalignment": "Mismatch between color-channel edge/gradient patterns.",
    "combined_z_mean": "Aggregate: mean of per-detector z-scores (higher => more suspicious on average).",
}


def _safe_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(v: object) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        f = _safe_float(s)
        return int(f) if f is not None else None


def _read_csv_rows(path: Path, max_rows: int = 0) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for idx, row in enumerate(reader):
            if max_rows > 0 and idx >= max_rows:
                break
            rows.append({k: (row.get(k) or "") for k in fieldnames})
    return rows, fieldnames


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_latest_eval_root(runs_dir: Path) -> Path:
    candidates = sorted(p for p in runs_dir.glob("eval_transfer_*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No eval_transfer_* directories under: {runs_dir}")
    return candidates[-1]


def _split_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _pick_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except Exception:
        return draw.textsize(text, font=font)

def _multiline_text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, spacing: int = 2
) -> Tuple[int, int]:
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except Exception:
        return _text_size(draw, text, font)


def _wrap_label(s: str, max_line: int = 12) -> str:
    s = s.replace("combined_z_mean", "combined_z_mean")
    parts = s.split("_")
    lines: List[str] = []
    cur = ""
    for p in parts:
        if not cur:
            cur = p
            continue
        if len(cur) + 1 + len(p) <= max_line:
            cur = f"{cur}_{p}"
        else:
            lines.append(cur)
            cur = p
    if cur:
        lines.append(cur)
    if len(lines) > 3:
        lines = lines[:3]
        lines[-1] = lines[-1][: max(4, max_line - 1)] + "…"
    return "\n".join(lines)


def _short_run_name(name: str, max_len: int = 28) -> str:
    if len(name) <= max_len:
        return name
    # Keep timestamp prefix + last token for readability.
    parts = name.split("_")
    if len(parts) >= 3:
        head = "_".join(parts[:2])
        tail = parts[-1]
        mid = "…"
        candidate = f"{head}_{mid}_{tail}"
        if len(candidate) <= max_len:
            return candidate
    return name[: max_len - 1] + "…"


@dataclass(frozen=True)
class Sides:
    run_a: str
    run_b: str
    baseline_side: str  # "a" or "b"

    @property
    def baseline(self) -> str:
        return self.run_a if self.baseline_side == "a" else self.run_b

    @property
    def ours(self) -> str:
        return self.run_b if self.baseline_side == "a" else self.run_a

    def baseline_is_a(self) -> bool:
        return self.baseline_side == "a"


def _auto_baseline_side(
    run_a: str, run_b: str, baseline_substrs: Sequence[str], ours_substrs: Sequence[str]
) -> str:
    a = run_a.lower()
    b = run_b.lower()
    base_hits_a = any(s.lower() in a for s in baseline_substrs)
    base_hits_b = any(s.lower() in b for s in baseline_substrs)
    ours_hits_a = any(s.lower() in a for s in ours_substrs)
    ours_hits_b = any(s.lower() in b for s in ours_substrs)

    if base_hits_a and not base_hits_b:
        return "a"
    if base_hits_b and not base_hits_a:
        return "b"
    if ours_hits_a and not ours_hits_b:
        return "b"
    if ours_hits_b and not ours_hits_a:
        return "a"
    return "a"


def _discover_comparisons(eval_root: Path, only: Sequence[str]) -> List[Path]:
    only_set = {x for x in only if x}
    out: List[Path] = []
    for p in sorted(eval_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "viz":
            continue
        if only_set and p.name not in only_set:
            continue
        if (p / "deepfake").is_dir() or (p / "fidelity").is_dir():
            out.append(p)
    return out


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _draw_grouped_bar_chart(
    out_path: Path,
    title: str,
    categories: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], RGB]],
    y_label: str,
    footnote: str = "",
) -> None:
    n_cat = max(1, len(categories))
    n_series = max(1, len(series))

    width = max(960, 220 * n_cat + 320)
    height = 620
    margin_l, margin_r, margin_t, margin_b = 110, 30, 70, 190
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    if plot_w < 200 or plot_h < 200:
        raise ValueError("Plot area too small.")

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_title = _pick_font(18)
    font_axis = _pick_font(14)
    font_tick = _pick_font(13)

    # Title
    tw, th = _text_size(draw, title, font_title)
    draw.text(((width - tw) // 2, 18), title, fill=(0, 0, 0), font=font_title)

    # Collect values
    values: List[float] = []
    for _, vals, _ in series:
        for v in vals:
            if math.isfinite(float(v)):
                values.append(float(v))
    if not values:
        values = [0.0]

    vmin = min(values + [0.0])
    vmax = max(values + [0.0])
    if math.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0
    pad = 0.08 * (vmax - vmin)
    vmin -= pad
    vmax += pad

    def y_of(v: float) -> int:
        t = (v - vmin) / (vmax - vmin)
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return int(margin_t + (1.0 - t) * plot_h)

    x0, y0 = margin_l, margin_t
    x1, y1 = margin_l + plot_w, margin_t + plot_h

    # Grid + ticks
    tick_count = 5
    for i in range(tick_count + 1):
        v = vmin + (vmax - vmin) * i / tick_count
        yy = y_of(v)
        draw.line([(x0, yy), (x1, yy)], fill=(230, 230, 230), width=1)
        label = f"{v:.3g}"
        lw, lh = _text_size(draw, label, font_tick)
        draw.text((x0 - 10 - lw, yy - lh // 2), label, fill=(0, 0, 0), font=font_tick)

    # Axes
    draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 0, 0), width=2)
    y_zero = y_of(0.0)
    if y0 <= y_zero <= y1:
        draw.line([(x0, y_zero), (x1, y_zero)], fill=(0, 0, 0), width=2)

    # Y label
    yl = y_label
    lw, lh = _text_size(draw, yl, font_axis)
    draw.text((x0, y0 - lh - 6), yl, fill=(0, 0, 0), font=font_axis)

    # Bars
    group_w = plot_w / n_cat
    inner_pad = 0.12 * group_w
    bar_area_w = group_w - inner_pad
    bar_w = bar_area_w / n_series

    for ci, cat in enumerate(categories):
        gx0 = x0 + ci * group_w + inner_pad / 2
        for si, (_, vals, color) in enumerate(series):
            v = float(vals[ci]) if ci < len(vals) else math.nan
            if not math.isfinite(v):
                continue
            bx0 = gx0 + si * bar_w + 0.10 * bar_w
            bx1 = gx0 + (si + 1) * bar_w - 0.10 * bar_w
            yv = y_of(v)
            if v >= 0:
                top, bot = yv, y_zero
            else:
                top, bot = y_zero, yv
            draw.rectangle([(bx0, top), (bx1, bot)], fill=color, outline=(0, 0, 0), width=1)

        # Category label
        label = _wrap_label(cat)
        lw, lh = _multiline_text_size(draw, label, font_tick, spacing=2)
        cx = x0 + (ci + 0.5) * group_w
        draw.multiline_text(
            (int(cx - lw / 2), y1 + 8),
            label,
            fill=(0, 0, 0),
            font=font_tick,
            align="center",
            spacing=2,
        )

    # Legend
    leg_x = x0 + plot_w - 10
    leg_y = 18
    for name, _, color in reversed(series):
        label = name
        lw, lh = _text_size(draw, label, font_tick)
        leg_x0 = leg_x - (lw + 36)
        draw.rectangle([(leg_x0, leg_y + 3), (leg_x0 + 18, leg_y + 15)], fill=color, outline=(0, 0, 0))
        draw.text((leg_x0 + 24, leg_y), label, fill=(0, 0, 0), font=font_tick)
        leg_y += lh + 8

    if footnote:
        fn_font = _pick_font(12)
        fw, fh = _text_size(draw, footnote, fn_font)
        draw.text((x0, height - fh - 10), footnote, fill=(60, 60, 60), font=fn_font)

    _ensure_dir(out_path.parent)
    img.save(out_path)


def _draw_signed_bar_chart(
    out_path: Path,
    title: str,
    categories: Sequence[str],
    values: Sequence[float],
    y_label: str,
    footnote: str = "",
    pos_color: RGB = (46, 125, 50),
    neg_color: RGB = (198, 40, 40),
) -> None:
    colors = [(pos_color if (v is not None and float(v) >= 0) else neg_color) for v in values]
    series = [("value", list(values), (120, 120, 120))]

    # Use the grouped bar chart renderer, but draw per-bar colors ourselves.
    n_cat = max(1, len(categories))
    width = max(960, 220 * n_cat + 320)
    height = 620
    margin_l, margin_r, margin_t, margin_b = 110, 30, 70, 190
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_title = _pick_font(18)
    font_axis = _pick_font(14)
    font_tick = _pick_font(13)

    tw, th = _text_size(draw, title, font_title)
    draw.text(((width - tw) // 2, 18), title, fill=(0, 0, 0), font=font_title)

    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        vals = [0.0]
    vmin = min(vals + [0.0])
    vmax = max(vals + [0.0])
    if math.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0
    pad = 0.08 * (vmax - vmin)
    vmin -= pad
    vmax += pad

    def y_of(v: float) -> int:
        t = (v - vmin) / (vmax - vmin)
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return int(margin_t + (1.0 - t) * plot_h)

    x0, y0 = margin_l, margin_t
    x1, y1 = margin_l + plot_w, margin_t + plot_h

    tick_count = 5
    for i in range(tick_count + 1):
        v = vmin + (vmax - vmin) * i / tick_count
        yy = y_of(v)
        draw.line([(x0, yy), (x1, yy)], fill=(230, 230, 230), width=1)
        label = f"{v:.3g}"
        lw, lh = _text_size(draw, label, font_tick)
        draw.text((x0 - 10 - lw, yy - lh // 2), label, fill=(0, 0, 0), font=font_tick)

    draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 0, 0), width=2)
    y_zero = y_of(0.0)
    if y0 <= y_zero <= y1:
        draw.line([(x0, y_zero), (x1, y_zero)], fill=(0, 0, 0), width=2)

    lw, lh = _text_size(draw, y_label, font_axis)
    draw.text((x0, y0 - lh - 6), y_label, fill=(0, 0, 0), font=font_axis)

    group_w = plot_w / n_cat
    inner_pad = 0.18 * group_w
    bar_w = group_w - inner_pad
    for ci, cat in enumerate(categories):
        v = float(values[ci]) if ci < len(values) else math.nan
        if not math.isfinite(v):
            continue
        bx0 = x0 + ci * group_w + inner_pad / 2
        bx1 = bx0 + bar_w
        yv = y_of(v)
        if v >= 0:
            top, bot = yv, y_zero
        else:
            top, bot = y_zero, yv
        draw.rectangle([(bx0, top), (bx1, bot)], fill=colors[ci], outline=(0, 0, 0), width=1)

        label = _wrap_label(cat)
        lw, lh = _multiline_text_size(draw, label, font_tick, spacing=2)
        cx = x0 + (ci + 0.5) * group_w
        draw.multiline_text(
            (int(cx - lw / 2), y1 + 8),
            label,
            fill=(0, 0, 0),
            font=font_tick,
            align="center",
            spacing=2,
        )

    if footnote:
        fn_font = _pick_font(12)
        fw, fh = _text_size(draw, footnote, fn_font)
        draw.text((x0, height - fh - 10), footnote, fill=(60, 60, 60), font=fn_font)

    _ensure_dir(out_path.parent)
    img.save(out_path)


def _render_histogram(
    title: str,
    values: Sequence[float],
    x_label: str,
    bins: int = 40,
    width: int = 520,
    height: int = 360,
) -> Image.Image:
    vals = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        vals = np.asarray([0.0], dtype=np.float64)

    hist, edges = np.histogram(vals, bins=bins)
    hist = hist.astype(np.float64)
    max_h = float(np.max(hist)) if hist.size else 1.0

    margin_l, margin_r, margin_t, margin_b = 70, 20, 48, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_title = _pick_font(14)
    font_tick = _pick_font(12)
    font_axis = _pick_font(12)

    draw.text((10, 10), title, fill=(0, 0, 0), font=font_title)

    x0, y0 = margin_l, margin_t
    x1, y1 = margin_l + plot_w, margin_t + plot_h
    draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 0, 0), width=2)

    # X range
    xmin = float(edges[0])
    xmax = float(edges[-1])
    if math.isclose(xmin, xmax):
        xmin -= 1.0
        xmax += 1.0

    def x_of(v: float) -> int:
        t = (v - xmin) / (xmax - xmin)
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return int(x0 + t * plot_w)

    def y_of_count(c: float) -> int:
        t = c / max(1.0, max_h)
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return int(y1 - t * plot_h)

    # Grid + y ticks
    for i in range(5):
        c = max_h * i / 4.0
        yy = y_of_count(c)
        draw.line([(x0, yy), (x1, yy)], fill=(230, 230, 230), width=1)
        label = f"{int(round(c))}"
        lw, lh = _text_size(draw, label, font_tick)
        draw.text((x0 - 8 - lw, yy - lh // 2), label, fill=(0, 0, 0), font=font_tick)

    # Bars
    for i in range(hist.size):
        left = x_of(float(edges[i]))
        right = x_of(float(edges[i + 1]))
        top = y_of_count(float(hist[i]))
        draw.rectangle([(left, top), (right, y1)], fill=(120, 170, 255), outline=(0, 0, 0))

    # Zero line
    if xmin <= 0.0 <= xmax:
        zx = x_of(0.0)
        draw.line([(zx, y0), (zx, y1)], fill=(0, 0, 0), width=2)
        draw.text((zx + 4, y0 + 4), "0", fill=(0, 0, 0), font=font_tick)

    # X label + min/max
    lw, lh = _text_size(draw, x_label, font_axis)
    draw.text((x0 + (plot_w - lw) // 2, y1 + 8), x_label, fill=(0, 0, 0), font=font_axis)
    draw.text((x0, y1 + 8), f"{xmin:.2g}", fill=(60, 60, 60), font=font_tick)
    rw, rh = _text_size(draw, f"{xmax:.2g}", font_tick)
    draw.text((x1 - rw, y1 + 8), f"{xmax:.2g}", fill=(60, 60, 60), font=font_tick)

    return img


def _tile_images(images: Sequence[Image.Image], cols: int, pad: int = 16, bg: RGB = (255, 255, 255)) -> Image.Image:
    if not images:
        return Image.new("RGB", (10, 10), bg)
    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    w = max(im.size[0] for im in images)
    h = max(im.size[1] for im in images)
    out = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (w + pad) + (w - im.size[0]) // 2
        y = pad + r * (h + pad) + (h - im.size[1]) // 2
        out.paste(im, (x, y))
    return out


def _fit_into(img: Image.Image, size: Tuple[int, int], bg: RGB = (245, 245, 245)) -> Image.Image:
    w, h = img.size
    tw, th = size
    if w <= 0 or h <= 0:
        return Image.new("RGB", size, bg)
    s = min(tw / w, th / h)
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    resized = img.resize((nw, nh), Image.BILINEAR)
    out = Image.new("RGB", (tw, th), bg)
    out.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return out


def _load_rgb(path: str) -> Optional[Image.Image]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            return im.convert("RGB")
    except Exception:
        return None


def _make_montage(
    out_path: Path,
    title: str,
    rows: Sequence[Tuple[str, float, str, str, str, str]],
    # (pair_id, score, person_path, cloth_path, baseline_pred_path, ours_pred_path)
    baseline_label: str,
    ours_label: str,
    cell_size: Tuple[int, int] = (220, 280),
) -> None:
    _ensure_dir(out_path.parent)

    font_title = _pick_font(18)
    font_hdr = _pick_font(14)
    font_row = _pick_font(12)

    pad = 12
    header_h = 74
    row_h = cell_size[1] + 34
    left_w = 260

    col_titles = ["person", "cloth", baseline_label, ours_label]
    cols = len(col_titles)
    width = left_w + cols * cell_size[0] + (cols + 1) * pad
    height = header_h + len(rows) * row_h + pad

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    tw, th = _text_size(draw, title, font_title)
    draw.text(((width - tw) // 2, 10), title, fill=(0, 0, 0), font=font_title)

    # Column headers
    y_hdr = 10 + th + 14
    for i, t in enumerate(col_titles):
        x = left_w + pad + i * (cell_size[0] + pad)
        draw.rectangle([(x, y_hdr), (x + cell_size[0], y_hdr + 24)], fill=(240, 240, 240), outline=(0, 0, 0))
        lw, lh = _text_size(draw, t, font_hdr)
        draw.text((x + (cell_size[0] - lw) // 2, y_hdr + (24 - lh) // 2), t, fill=(0, 0, 0), font=font_hdr)

    # Rows
    y = header_h
    for pair_id, score, person_p, cloth_p, base_p, ours_p in rows:
        # Left text
        draw.rectangle([(pad, y), (left_w - pad, y + row_h - pad)], fill=(250, 250, 250), outline=(0, 0, 0))
        txt = f"{pair_id}\nscore: {score:+.4f}"
        draw.multiline_text((pad + 10, y + 10), txt, fill=(0, 0, 0), font=font_row, spacing=4)

        paths = [person_p, cloth_p, base_p, ours_p]
        for i, pth in enumerate(paths):
            x = left_w + pad + i * (cell_size[0] + pad)
            cell = Image.new("RGB", cell_size, (245, 245, 245))
            src = _load_rgb(pth)
            if src is None:
                d2 = ImageDraw.Draw(cell)
                d2.text((10, 10), "missing", fill=(150, 0, 0), font=font_row)
            else:
                cell = _fit_into(src, cell_size)
            img.paste(cell, (x, y))
            draw.rectangle([(x, y), (x + cell_size[0], y + cell_size[1])], outline=(0, 0, 0), width=2)
        y += row_h

    img.save(out_path)


def _find_overlap_csv(eval_root: Path, comparison_name: str) -> Optional[Path]:
    direct = eval_root / f"overlap_{comparison_name}.csv"
    if direct.exists():
        return direct
    matches = sorted(eval_root.glob(f"overlap*{comparison_name}*.csv"))
    return matches[0] if matches else None


def _build_overlap_index(overlap_csv: Path, sides: Sides) -> Dict[str, Dict[str, str]]:
    rows, cols = _read_csv_rows(overlap_csv)
    pred_a_col = f"pred_path_{sides.run_a}"
    pred_b_col = f"pred_path_{sides.run_b}"
    cloth_a_col = f"cloth_path_{sides.run_a}"
    cloth_b_col = f"cloth_path_{sides.run_b}"
    gt_a_col = f"gt_path_{sides.run_a}"
    gt_b_col = f"gt_path_{sides.run_b}"

    idx: Dict[str, Dict[str, str]] = {}
    for r in rows:
        pid = r.get("pair_id", "")
        if not pid:
            continue
        idx[pid] = {
            "person_path": r.get("person_path", ""),
            "pred_a": r.get(pred_a_col, ""),
            "pred_b": r.get(pred_b_col, ""),
            "cloth_a": r.get(cloth_a_col, r.get("cloth_path", "")),
            "cloth_b": r.get(cloth_b_col, r.get("cloth_path", "")),
            "gt_a": r.get(gt_a_col, r.get("gt_path", "")),
            "gt_b": r.get(gt_b_col, r.get("gt_path", "")),
        }
    return idx


def _build_pair_metrics(
    comp_dir: Path, sides: Sides, max_pairs: int = 0
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Return per-pair deltas.

    Outputs (all keyed by pair_id):
      - deepfake_improvement: baseline - ours  (positive means ours less fake)
      - fidelity_improvement: ours - baseline  (positive means ours more faithful)
      - fidelity_avg_a: mean(cloth/face/pose) for run_a
      - fidelity_avg_b: mean(cloth/face/pose) for run_b
    """
    deepfake_comb = comp_dir / "deepfake" / "deepfake_combined_per_pair.csv"
    fidelity_pp = comp_dir / "fidelity" / "fidelity_per_pair.csv"
    if not deepfake_comb.exists() or not fidelity_pp.exists():
        return {}, {}, {}, {}

    deep_rows, _ = _read_csv_rows(deepfake_comb, max_rows=max_pairs)
    fid_rows, _ = _read_csv_rows(fidelity_pp, max_rows=max_pairs)

    deep_a: Dict[str, float] = {}
    deep_b: Dict[str, float] = {}
    for r in deep_rows:
        pid = r.get("pair_id", "")
        if not pid:
            continue
        a = _safe_float(r.get("combined_z_a"))
        b = _safe_float(r.get("combined_z_b"))
        if a is None or b is None:
            continue
        deep_a[pid] = float(a)
        deep_b[pid] = float(b)

    fid_a: Dict[str, float] = {}
    fid_b: Dict[str, float] = {}
    for r in fid_rows:
        pid = r.get("pair_id", "")
        if not pid:
            continue
        ca = _safe_float(r.get("cloth_score_a"))
        cb = _safe_float(r.get("cloth_score_b"))
        fa = _safe_float(r.get("face_score_a"))
        fb = _safe_float(r.get("face_score_b"))
        pa = _safe_float(r.get("pose_score_a"))
        pb = _safe_float(r.get("pose_score_b"))
        if None in (ca, cb, fa, fb, pa, pb):
            continue
        fid_a[pid] = float((ca + fa + pa) / 3.0)
        fid_b[pid] = float((cb + fb + pb) / 3.0)

    deep_impr: Dict[str, float] = {}
    fid_impr: Dict[str, float] = {}

    for pid in set(deep_a) & set(deep_b) & set(fid_a) & set(fid_b):
        if sides.baseline_is_a():
            base_deep, ours_deep = deep_a[pid], deep_b[pid]
            base_fid, ours_fid = fid_a[pid], fid_b[pid]
        else:
            base_deep, ours_deep = deep_b[pid], deep_a[pid]
            base_fid, ours_fid = fid_b[pid], fid_a[pid]

        deep_impr[pid] = float(base_deep - ours_deep)  # lower is better
        fid_impr[pid] = float(ours_fid - base_fid)  # higher is better

    return deep_impr, fid_impr, fid_a, fid_b


def _is_tie(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def _deepfake_win_tie_rates(comp_dir: Path, sides: Sides) -> Dict[str, Dict[str, float]]:
    """Compute strict win/tie rates from per-pair CSVs.

    Win definition (deepfake): ours is better if score_ours < score_baseline.
    Returns: detector -> {ours_win_rate, tie_rate, n}
    """
    out: Dict[str, Dict[str, float]] = {}

    per_pair = comp_dir / "deepfake" / "deepfake_per_pair.csv"
    if per_pair.exists():
        rows, _ = _read_csv_rows(per_pair)
        wins: Dict[str, int] = {}
        ties: Dict[str, int] = {}
        totals: Dict[str, int] = {}
        for r in rows:
            det = (r.get("detector") or "").strip()
            if not det:
                continue
            sa = _safe_float(r.get("score_a"))
            sb = _safe_float(r.get("score_b"))
            if sa is None or sb is None:
                continue
            base = sa if sides.baseline_is_a() else sb
            ours = sb if sides.baseline_is_a() else sa
            totals[det] = totals.get(det, 0) + 1
            if _is_tie(ours, base):
                ties[det] = ties.get(det, 0) + 1
            elif ours < base:
                wins[det] = wins.get(det, 0) + 1

        for det, n in totals.items():
            if n <= 0:
                continue
            out[det] = {
                "ours_win_rate": float(wins.get(det, 0) / n),
                "tie_rate": float(ties.get(det, 0) / n),
                "n": float(n),
            }

    combined = comp_dir / "deepfake" / "deepfake_combined_per_pair.csv"
    if combined.exists():
        rows, _ = _read_csv_rows(combined)
        win = tie = total = 0
        for r in rows:
            za = _safe_float(r.get("combined_z_a"))
            zb = _safe_float(r.get("combined_z_b"))
            if za is None or zb is None:
                continue
            base = za if sides.baseline_is_a() else zb
            ours = zb if sides.baseline_is_a() else za
            total += 1
            if _is_tie(ours, base):
                tie += 1
            elif ours < base:
                win += 1
        if total > 0:
            out["combined_z_mean"] = {
                "ours_win_rate": float(win / total),
                "tie_rate": float(tie / total),
                "n": float(total),
            }

    return out


def _fidelity_win_tie_rates(comp_dir: Path, sides: Sides) -> Dict[str, Dict[str, float]]:
    """Compute strict win/tie rates from per-pair CSVs.

    Win definition (fidelity): ours is better if score_ours > score_baseline.
    Returns: metric -> {ours_win_rate, tie_rate, n}
    """
    per_pair = comp_dir / "fidelity" / "fidelity_per_pair.csv"
    if not per_pair.exists():
        return {}

    rows, _ = _read_csv_rows(per_pair)
    metrics = ["cloth_score", "face_score", "pose_score"]
    wins: Dict[str, int] = {m: 0 for m in metrics}
    ties: Dict[str, int] = {m: 0 for m in metrics}
    totals: Dict[str, int] = {m: 0 for m in metrics}

    for r in rows:
        for m in metrics:
            a = _safe_float(r.get(f"{m}_a"))
            b = _safe_float(r.get(f"{m}_b"))
            if a is None or b is None:
                continue
            base = a if sides.baseline_is_a() else b
            ours = b if sides.baseline_is_a() else a
            totals[m] += 1
            if _is_tie(ours, base):
                ties[m] += 1
            elif ours > base:
                wins[m] += 1

    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        n = totals[m]
        if n <= 0:
            continue
        out[m] = {
            "ours_win_rate": float(wins[m] / n),
            "tie_rate": float(ties[m] / n),
            "n": float(n),
        }
    return out


def _draw_scatter(
    out_path: Path,
    title: str,
    x: Sequence[float],
    y: Sequence[float],
    x_label: str,
    y_label: str,
) -> None:
    _ensure_dir(out_path.parent)

    xs = np.asarray([float(v) for v in x if v is not None and math.isfinite(float(v))], dtype=np.float64)
    ys = np.asarray([float(v) for v in y if v is not None and math.isfinite(float(v))], dtype=np.float64)
    n = min(xs.size, ys.size)
    if n == 0:
        xs = np.asarray([0.0], dtype=np.float64)
        ys = np.asarray([0.0], dtype=np.float64)
        n = 1
    else:
        xs = xs[:n]
        ys = ys[:n]

    width, height = 820, 560
    margin_l, margin_r, margin_t, margin_b = 90, 30, 60, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    if math.isclose(xmin, xmax):
        xmin -= 1.0
        xmax += 1.0
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    xpad = 0.08 * (xmax - xmin)
    ypad = 0.08 * (ymax - ymin)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    def x_of(v: float) -> int:
        t = (v - xmin) / (xmax - xmin)
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return int(margin_l + t * plot_w)

    def y_of(v: float) -> int:
        t = (v - ymin) / (ymax - ymin)
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return int(margin_t + (1.0 - t) * plot_h)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_title = _pick_font(16)
    font_tick = _pick_font(12)
    font_axis = _pick_font(13)

    tw, th = _text_size(draw, title, font_title)
    draw.text(((width - tw) // 2, 16), title, fill=(0, 0, 0), font=font_title)

    x0, y0 = margin_l, margin_t
    x1, y1 = margin_l + plot_w, margin_t + plot_h
    draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 0, 0), width=2)

    # grid
    for i in range(1, 5):
        xx = x0 + int(plot_w * i / 5)
        yy = y0 + int(plot_h * i / 5)
        draw.line([(xx, y0), (xx, y1)], fill=(235, 235, 235), width=1)
        draw.line([(x0, yy), (x1, yy)], fill=(235, 235, 235), width=1)

    # zero axes
    if xmin <= 0.0 <= xmax:
        zx = x_of(0.0)
        draw.line([(zx, y0), (zx, y1)], fill=(0, 0, 0), width=2)
    if ymin <= 0.0 <= ymax:
        zy = y_of(0.0)
        draw.line([(x0, zy), (x1, zy)], fill=(0, 0, 0), width=2)

    # points
    for xi, yi in zip(xs, ys, strict=False):
        px = x_of(float(xi))
        py = y_of(float(yi))
        r = 3
        draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=(30, 90, 200), outline=(0, 0, 0))

    # axis labels
    lw, lh = _text_size(draw, x_label, font_axis)
    draw.text((x0 + (plot_w - lw) // 2, y1 + 16), x_label, fill=(0, 0, 0), font=font_axis)
    lw, lh = _text_size(draw, y_label, font_axis)
    draw.text((x0, y0 - lh - 10), y_label, fill=(0, 0, 0), font=font_axis)

    # min/max tick labels
    draw.text((x0, y1 + 16), f"{xmin:.3g}", fill=(60, 60, 60), font=font_tick)
    rw, rh = _text_size(draw, f"{xmax:.3g}", font_tick)
    draw.text((x1 - rw, y1 + 16), f"{xmax:.3g}", fill=(60, 60, 60), font=font_tick)
    draw.text((x0 - 65, y1 - 12), f"{ymin:.3g}", fill=(60, 60, 60), font=font_tick)
    draw.text((x0 - 65, y0 - 6), f"{ymax:.3g}", fill=(60, 60, 60), font=font_tick)

    # correlation
    if xs.size >= 3:
        sx = float(np.std(xs))
        sy = float(np.std(ys))
        if sx > 1e-12 and sy > 1e-12:
            r = float(np.corrcoef(xs, ys)[0, 1])
            if math.isfinite(r):
                txt = f"corr(x,y) = {r:+.3f}  (n={int(xs.size)})"
                draw.text((x0 + 10, y0 + 10), txt, fill=(0, 0, 0), font=font_tick)

    img.save(out_path)


def _format_float(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return ""
    return f"{v:.{digits}f}"

def _format_pct(v: Optional[float], digits: int = 1) -> str:
    if v is None:
        return ""
    return f"{100.0 * v:.{digits}f}%"


def _write_report(path: Path, lines: Sequence[str]) -> None:
    _ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _visualize_comparison(
    eval_root: Path,
    comp_dir: Path,
    out_dir: Path,
    sides: Sides,
    top_k: int,
    max_pairs: int,
    make_montage: bool,
) -> List[str]:
    """Returns markdown lines for report section."""
    comp_name = comp_dir.name

    deep_sum_csv = comp_dir / "deepfake" / "deepfake_summary.csv"
    deep_sum_json = comp_dir / "deepfake" / "deepfake_summary.json"
    fid_sum_csv = comp_dir / "fidelity" / "fidelity_summary.csv"
    fid_sum_json = comp_dir / "fidelity" / "fidelity_summary.json"

    lines: List[str] = []
    lines.append(f"## {comp_name}")
    lines.append("")
    lines.append(f"- run_a: `{sides.run_a}`")
    lines.append(f"- run_b: `{sides.run_b}`")
    lines.append(f"- baseline: `{sides.baseline}`")
    lines.append(f"- ours: `{sides.ours}`")
    lines.append("")

    baseline_label = _short_run_name(sides.baseline)
    ours_label = _short_run_name(sides.ours)

    # Deepfake plots + table
    if deep_sum_csv.exists():
        deep_rows, _ = _read_csv_rows(deep_sum_csv)
        deep_meta: Dict[str, object] = _load_json(deep_sum_json) if deep_sum_json.exists() else {}
        dets = [r.get("detector", "") for r in deep_rows if r.get("detector", "")]
        a_means = [_safe_float(r.get("run_a_mean")) or math.nan for r in deep_rows]
        b_means = [_safe_float(r.get("run_b_mean")) or math.nan for r in deep_rows]
        if sides.baseline_is_a():
            base_means, ours_means = a_means, b_means
        else:
            base_means, ours_means = b_means, a_means
        improvements = [float(bm - om) for bm, om in zip(base_means, ours_means, strict=False)]

        deep_means_png = out_dir / "deepfake_means.png"
        deep_impr_png = out_dir / "deepfake_improvement.png"
        deep_flag_png = out_dir / "deepfake_flag_rates.png"
        _draw_grouped_bar_chart(
            deep_means_png,
            title="Deepfake detector scores (lower is better)",
            categories=dets,
            series=[
                (f"baseline", base_means, (66, 133, 244)),
                (f"ours", ours_means, (244, 180, 0)),
            ],
            y_label="score",
            footnote="Scores are detector-specific; compare within each detector.",
        )
        _draw_signed_bar_chart(
            deep_impr_png,
            title="Deepfake improvement (baseline - ours)  [positive is better]",
            categories=dets,
            values=improvements,
            y_label="improvement",
        )

        a_flags = [_safe_float(r.get("run_a_flag_rate")) for r in deep_rows]
        b_flags = [_safe_float(r.get("run_b_flag_rate")) for r in deep_rows]
        if sides.baseline_is_a():
            base_flags, ours_flags = a_flags, b_flags
        else:
            base_flags, ours_flags = b_flags, a_flags
        _draw_grouped_bar_chart(
            deep_flag_png,
            title="Deepfake flag rate (fraction score > threshold)",
            categories=dets,
            series=[
                ("baseline", [v if v is not None else math.nan for v in base_flags], (66, 133, 244)),
                ("ours", [v if v is not None else math.nan for v in ours_flags], (244, 180, 0)),
            ],
            y_label="flag_rate",
            footnote="Threshold is detector-specific (see table). Lower is better.",
        )

        lines.append("### Deepfake")
        lines.append("")
        lines.append(f"![]({comp_name}/deepfake_means.png)")
        lines.append(f"![]({comp_name}/deepfake_improvement.png)")
        lines.append(f"![]({comp_name}/deepfake_flag_rates.png)")
        lines.append("")

        det_list = [str(x) for x in (deep_meta.get("detectors") or []) if str(x)]
        if det_list:
            lines.append("Detectors:")
            for d in det_list:
                desc = DETECTOR_DESCRIPTIONS.get(d, "")
                suffix = f" — {desc}" if desc else ""
                lines.append(f"- `{d}`{suffix}")
            if "combined_z_mean" in dets:
                desc = DETECTOR_DESCRIPTIONS.get("combined_z_mean", "")
                suffix = f" — {desc}" if desc else ""
                lines.append(f"- `combined_z_mean`{suffix}")
            lines.append("")

        pair_rates = _deepfake_win_tie_rates(comp_dir, sides)

        lines.append("Table (lower score & lower flag_rate are better):")
        lines.append("")
        lines.append(
            "| detector | n | baseline_mean | ours_mean | mean(b-o) | median(b-o) | baseline_flag | ours_flag | ours_better_win | tie | threshold | threshold_source | p |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
        for r, bm, om, imp in zip(deep_rows, base_means, ours_means, improvements, strict=False):
            det = r.get("detector", "")
            n = _safe_int(r.get("n_pairs")) or 0
            p = _safe_float(r.get("paired_wilcoxon_p"))
            mean_ab = _safe_float(r.get("mean_delta_a_minus_b"))
            med_ab = _safe_float(r.get("median_delta_a_minus_b"))
            if sides.baseline_is_a():
                mean_bo = mean_ab
                med_bo = med_ab
                base_flag = _safe_float(r.get("run_a_flag_rate"))
                ours_flag = _safe_float(r.get("run_b_flag_rate"))
                a_more_fake = _safe_float(r.get("a_more_fake_win_rate"))
                ours_better = a_more_fake
            else:
                mean_bo = (-mean_ab) if mean_ab is not None else None
                med_bo = (-med_ab) if med_ab is not None else None
                base_flag = _safe_float(r.get("run_b_flag_rate"))
                ours_flag = _safe_float(r.get("run_a_flag_rate"))
                a_more_fake = _safe_float(r.get("a_more_fake_win_rate"))
                ours_better = (1.0 - a_more_fake) if a_more_fake is not None else None

            tie_rate = None
            pair_stat = pair_rates.get(det)
            if pair_stat is not None:
                ours_better = float(pair_stat.get("ours_win_rate", ours_better or 0.0))
                tie_rate = float(pair_stat.get("tie_rate", 0.0))

            thr = _safe_float(r.get("threshold"))
            thr_src = (r.get("threshold_source") or "").strip()
            lines.append(
                "| `{det}` | {n} | {bm} | {om} | {mean_bo} | {med_bo} | {bflag} | {oflag} | {win} | {tie} | {thr} | `{thr_src}` | {p} |".format(
                    det=det,
                    n=n,
                    bm=_format_float(bm),
                    om=_format_float(om),
                    mean_bo=_format_float(mean_bo),
                    med_bo=_format_float(med_bo),
                    bflag=_format_pct(base_flag, digits=1),
                    oflag=_format_pct(ours_flag, digits=1),
                    win=_format_pct(ours_better, digits=1),
                    tie=_format_pct(tie_rate, digits=1),
                    thr=_format_float(thr),
                    thr_src=thr_src or "",
                    p=_format_float(p, 6),
                )
            )
        lines.append("")
    else:
        lines.append("### Deepfake")
        lines.append("")
        lines.append("_No deepfake outputs found._")
        lines.append("")

    # Fidelity plots + table
    if fid_sum_csv.exists():
        fid_rows, _ = _read_csv_rows(fid_sum_csv)
        fid_meta: Dict[str, object] = _load_json(fid_sum_json) if fid_sum_json.exists() else {}
        metrics = [r.get("metric", "") for r in fid_rows if r.get("metric", "")]
        a_means = [_safe_float(r.get("run_a_mean")) or math.nan for r in fid_rows]
        b_means = [_safe_float(r.get("run_b_mean")) or math.nan for r in fid_rows]
        if sides.baseline_is_a():
            base_means, ours_means = a_means, b_means
        else:
            base_means, ours_means = b_means, a_means
        improvements = [float(om - bm) for bm, om in zip(base_means, ours_means, strict=False)]

        fid_means_png = out_dir / "fidelity_means.png"
        fid_impr_png = out_dir / "fidelity_improvement.png"
        _draw_grouped_bar_chart(
            fid_means_png,
            title="Fidelity scores (higher is better)",
            categories=metrics,
            series=[
                (f"baseline", base_means, (66, 133, 244)),
                (f"ours", ours_means, (244, 180, 0)),
            ],
            y_label="score",
            footnote="Scores are in [0,1] where possible (higher = more similar).",
        )
        _draw_signed_bar_chart(
            fid_impr_png,
            title="Fidelity improvement (ours - baseline)  [positive is better]",
            categories=metrics,
            values=improvements,
            y_label="improvement",
        )

        lines.append("### Fidelity")
        lines.append("")
        lines.append(f"![]({comp_name}/fidelity_means.png)")
        lines.append(f"![]({comp_name}/fidelity_improvement.png)")
        lines.append("")
        if fid_meta is not None:
            lines.append("Metrics:")
            lines.append("- `face_score`: face similarity (produced vs person).")
            lines.append("- `cloth_score`: cloth similarity (produced vs cloth reference).")
            lines.append("- `pose_score`: posture similarity (produced vs person).")
            lines.append("")

        lines.append("Table (higher is better):")
        lines.append("")
        pair_rates = _fidelity_win_tie_rates(comp_dir, sides)
        lines.append("| metric | n | baseline_mean | ours_mean | mean(o-b) | median(o-b) | ours_better_win | tie | p |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r, bm, om, imp in zip(fid_rows, base_means, ours_means, improvements, strict=False):
            met = r.get("metric", "")
            n = _safe_int(r.get("n_pairs")) or 0
            p = _safe_float(r.get("paired_wilcoxon_p"))
            mean_ab = _safe_float(r.get("mean_delta_a_minus_b"))
            med_ab = _safe_float(r.get("median_delta_a_minus_b"))
            a_win = _safe_float(r.get("a_better_win_rate"))
            if sides.baseline_is_a():
                mean_ob = (-mean_ab) if mean_ab is not None else None
                med_ob = (-med_ab) if med_ab is not None else None
                ours_win = (1.0 - a_win) if a_win is not None else None
            else:
                mean_ob = mean_ab
                med_ob = med_ab
                ours_win = a_win

            tie_rate = None
            pair_stat = pair_rates.get(met)
            if pair_stat is not None:
                ours_win = float(pair_stat.get("ours_win_rate", ours_win or 0.0))
                tie_rate = float(pair_stat.get("tie_rate", 0.0))

            lines.append(
                f"| `{met}` | {n} | {_format_float(bm)} | {_format_float(om)} | {_format_float(mean_ob)} | {_format_float(med_ob)} | {_format_pct(ours_win, digits=1)} | {_format_pct(tie_rate, digits=1)} | {_format_float(p, 6)} |"
            )
        lines.append("")
    else:
        lines.append("### Fidelity")
        lines.append("")
        lines.append("_No fidelity outputs found._")
        lines.append("")

    # Per-pair analyses: histograms, scatter, montages
    deep_impr, fid_impr, _, _ = _build_pair_metrics(comp_dir, sides, max_pairs=max_pairs)
    if deep_impr and fid_impr:
        # Scatter: fidelity vs deepfake improvements
        scatter_png = out_dir / "scatter_fidelity_vs_deepfake.png"
        _draw_scatter(
            scatter_png,
            title="Per-pair tradeoff",
            x=list(fid_impr.values()),
            y=list(deep_impr.values()),
            x_label="fidelity improvement (ours - baseline)",
            y_label="deepfake improvement (baseline - ours)",
        )

        # Histograms
        h1 = _render_histogram(
            "Fidelity improvement (ours - baseline)", list(fid_impr.values()), x_label="delta", bins=40
        )
        h2 = _render_histogram(
            "Deepfake improvement (baseline - ours)", list(deep_impr.values()), x_label="delta", bins=40
        )
        hist_grid = _tile_images([h1, h2], cols=2)
        hist_png = out_dir / "improvement_histograms.png"
        _ensure_dir(hist_png.parent)
        hist_grid.save(hist_png)

        lines.append("### Per-pair")
        lines.append("")
        lines.append(f"![]({comp_name}/improvement_histograms.png)")
        lines.append(f"![]({comp_name}/scatter_fidelity_vs_deepfake.png)")
        lines.append("")

        if make_montage:
            overlap_csv = _find_overlap_csv(eval_root, comp_name)
            if overlap_csv is not None and overlap_csv.exists():
                overlap_idx = _build_overlap_index(overlap_csv, sides)

                # Rank by improvements
                best_fid = sorted(fid_impr.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
                worst_fid = sorted(fid_impr.items(), key=lambda kv: kv[1])[:top_k]

                def _row(pid: str, score: float) -> Optional[Tuple[str, float, str, str, str, str]]:
                    r = overlap_idx.get(pid)
                    if not r:
                        return None
                    person = r.get("person_path", "")
                    cloth = r.get("cloth_a", "") or r.get("cloth_b", "")
                    if sides.baseline_is_a():
                        base_pred, ours_pred = r.get("pred_a", ""), r.get("pred_b", "")
                    else:
                        base_pred, ours_pred = r.get("pred_b", ""), r.get("pred_a", "")
                    return (pid, float(score), person, cloth, base_pred, ours_pred)

                best_rows = [x for x in (_row(pid, sc) for pid, sc in best_fid) if x is not None]
                worst_rows = [x for x in (_row(pid, sc) for pid, sc in worst_fid) if x is not None]

                if best_rows:
                    _make_montage(
                        out_dir / "top_fidelity_improvements.png",
                        title="Top fidelity improvements",
                        rows=best_rows,
                        baseline_label="baseline",
                        ours_label="ours",
                    )
                    lines.append(f"![]({comp_name}/top_fidelity_improvements.png)")
                    lines.append("")
                if worst_rows:
                    _make_montage(
                        out_dir / "worst_fidelity_improvements.png",
                        title="Worst fidelity improvements",
                        rows=worst_rows,
                        baseline_label="baseline",
                        ours_label="ours",
                    )
                    lines.append(f"![]({comp_name}/worst_fidelity_improvements.png)")
                    lines.append("")

            else:
                lines.append("_Overlap CSV not found; skipping montages._")
                lines.append("")

    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", type=Path, default=None, help="Path to runs/eval_transfer_YYYYMMDD_HHMMSS/")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Directory that contains eval_transfer_*/")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (default: <eval_root>/viz)")
    p.add_argument("--only", default="", help="Comma-separated comparison names to process (default: all).")
    p.add_argument(
        "--baseline-side",
        choices=["auto", "a", "b"],
        default="auto",
        help="Which side is baseline. Use 'auto' to infer from names.",
    )
    p.add_argument(
        "--baseline-substr",
        default="vitonhd_test",
        help="Comma-separated substrings that indicate a baseline run name (auto mode).",
    )
    p.add_argument(
        "--ours-substr",
        default="mytest",
        help="Comma-separated substrings that indicate an 'ours' run name (auto mode).",
    )
    p.add_argument("--top-k", type=int, default=12, help="How many examples to include in montages.")
    p.add_argument("--max-pairs", type=int, default=0, help="Cap per-pair reads for plots/montages (0 = all).")
    p.add_argument("--skip-montage", action="store_true", help="Skip example montages (faster).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    eval_root = args.eval_root.resolve() if args.eval_root else _maybe_latest_eval_root(runs_dir)
    if not eval_root.exists():
        raise FileNotFoundError(f"Eval root not found: {eval_root}")

    out_root = (args.outdir.resolve() if args.outdir else (eval_root / "viz")).resolve()
    _ensure_dir(out_root)

    only = _split_csv_list(args.only)
    comps = _discover_comparisons(eval_root, only)
    if not comps:
        raise RuntimeError(f"No comparison directories found under: {eval_root}")

    baseline_sub = _split_csv_list(args.baseline_substr)
    ours_sub = _split_csv_list(args.ours_substr)

    report: List[str] = []
    report.append("# Eval Transfer Visualization")
    report.append("")
    report.append(f"- eval_root: `{eval_root}`")
    report.append(f"- generated: `{out_root}`")
    report.append("")
    report.append("Notes:")
    report.append("- Deepfake detector scores: lower is better.")
    report.append("- Fidelity scores: higher is better.")
    report.append("")

    for comp_dir in comps:
        # Resolve run names
        deep_sum_json = comp_dir / "deepfake" / "deepfake_summary.json"
        fid_sum_json = comp_dir / "fidelity" / "fidelity_summary.json"
        meta: Dict[str, object] = {}
        if deep_sum_json.exists():
            meta = _load_json(deep_sum_json)
        elif fid_sum_json.exists():
            meta = _load_json(fid_sum_json)
        run_a = str(meta.get("run_a", "run_a"))
        run_b = str(meta.get("run_b", "run_b"))

        if args.baseline_side == "auto":
            baseline_side = _auto_baseline_side(run_a, run_b, baseline_sub, ours_sub)
        else:
            baseline_side = args.baseline_side

        sides = Sides(run_a=run_a, run_b=run_b, baseline_side=baseline_side)
        comp_out = out_root / comp_dir.name
        _ensure_dir(comp_out)

        section = _visualize_comparison(
            eval_root=eval_root,
            comp_dir=comp_dir,
            out_dir=comp_out,
            sides=sides,
            top_k=max(1, int(args.top_k)),
            max_pairs=int(args.max_pairs),
            make_montage=not bool(args.skip_montage),
        )
        report.extend(section)
        report.append("")

    _write_report(out_root / "report.md", report)
    print(f"Saved report: {out_root / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
