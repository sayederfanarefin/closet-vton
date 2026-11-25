#!/usr/bin/env python3
"""Compare cloth, face, and posture correctness for two VTON runs.

Input CSV should be produced by `scripts/build_overlap_manifest.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import wilcoxon


EPS = 1e-8

try:
    import mediapipe as mp  # type: ignore

    HAS_MEDIAPIPE = True
except Exception:
    mp = None
    HAS_MEDIAPIPE = False


@lru_cache(maxsize=512)
def _load_rgb(path_str: str) -> np.ndarray:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


@lru_cache(maxsize=512)
def _load_mask(path_str: str) -> np.ndarray:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing mask: {path}")
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return arr


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(
        np.float32
    )


def _resize_rgb(rgb: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    out = im.resize((w, h), Image.BILINEAR)
    return np.asarray(out, dtype=np.float32) / 255.0


def _resize_mask(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    out = im.resize((w, h), Image.NEAREST)
    return np.asarray(out, dtype=np.float32) / 255.0


def _hist_rgb(rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    feats = []
    for ch in range(3):
        h, _ = np.histogram(rgb[:, :, ch], bins=bins, range=(0.0, 1.0), density=False)
        h = h.astype(np.float32)
        h /= np.sum(h) + EPS
        feats.append(h)
    return np.concatenate(feats, axis=0)


def _hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    numer = float(np.sum(np.minimum(a, b)))
    denom = float(min(np.sum(a), np.sum(b))) + EPS
    return numer / denom


def _edge_dir_hist(gray: np.ndarray, bins: int = 18) -> np.ndarray:
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    h, _ = np.histogram(ang, bins=bins, range=(0.0, 1.0), weights=mag, density=False)
    h = h.astype(np.float32)
    h /= np.sum(h) + EPS
    return h


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    if a.size != b.size:
        return math.nan
    am = float(np.mean(a))
    bm = float(np.mean(b))
    asd = float(np.std(a))
    bsd = float(np.std(b))
    if asd < EPS or bsd < EPS:
        return 0.5
    corr = float(np.mean((a - am) * (b - bm)) / (asd * bsd))
    corr = max(-1.0, min(1.0, corr))
    return 0.5 * (corr + 1.0)


def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    if np.count_nonzero(m) < 64:
        return _ncc(a, b)
    av = a[m]
    bv = b[m]
    am = float(np.mean(av))
    bm = float(np.mean(bv))
    asd = float(np.std(av))
    bsd = float(np.std(bv))
    if asd < EPS or bsd < EPS:
        return 0.5
    corr = float(np.mean((av - am) * (bv - bm)) / (asd * bsd))
    corr = max(-1.0, min(1.0, corr))
    return 0.5 * (corr + 1.0)


def _fallback_face_crop(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    y0 = int(0.06 * h)
    y1 = int(0.45 * h)
    x0 = int(0.25 * w)
    x1 = int(0.75 * w)
    return rgb[y0:y1, x0:x1]


def _fallback_cloth_crop(pred_rgb: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    h, w = pred_rgb.shape[:2]
    if mask is not None:
        if mask.shape != (h, w):
            mask = _resize_mask(mask, (h, w))
        m = mask > 0.5
        if np.count_nonzero(m) >= 128:
            ys, xs = np.where(m)
            y0, y1 = int(np.min(ys)), int(np.max(ys))
            x0, x1 = int(np.min(xs)), int(np.max(xs))
            pad_y = int(0.05 * max(1, y1 - y0 + 1))
            pad_x = int(0.05 * max(1, x1 - x0 + 1))
            y0 = max(0, y0 - pad_y)
            y1 = min(h - 1, y1 + pad_y)
            x0 = max(0, x0 - pad_x)
            x1 = min(w - 1, x1 + pad_x)
            return pred_rgb[y0 : y1 + 1, x0 : x1 + 1]

    # fallback torso crop
    y0 = int(0.22 * h)
    y1 = int(0.82 * h)
    x0 = int(0.18 * w)
    x1 = int(0.82 * w)
    return pred_rgb[y0:y1, x0:x1]


class MediaPipeHelper:
    def __init__(self, enable_face: bool, enable_pose: bool):
        self.enable_face = enable_face and HAS_MEDIAPIPE
        self.enable_pose = enable_pose and HAS_MEDIAPIPE

        self._face = None
        self._pose = None
        if self.enable_face:
            self._face = mp.solutions.face_detection.FaceDetection(  # type: ignore[attr-defined]
                model_selection=1, min_detection_confidence=0.5
            )
        if self.enable_pose:
            self._pose = mp.solutions.pose.Pose(  # type: ignore[attr-defined]
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )

    def close(self) -> None:
        if self._face is not None:
            self._face.close()
        if self._pose is not None:
            self._pose.close()

    def face_crop(self, rgb: np.ndarray) -> np.ndarray:
        if self._face is None:
            return _fallback_face_crop(rgb)
        h, w = rgb.shape[:2]
        img_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        result = self._face.process(img_u8)
        if result.detections:
            box = result.detections[0].location_data.relative_bounding_box
            x0 = max(0, int(box.xmin * w))
            y0 = max(0, int(box.ymin * h))
            x1 = min(w, int((box.xmin + box.width) * w))
            y1 = min(h, int((box.ymin + box.height) * h))
            if x1 - x0 >= 20 and y1 - y0 >= 20:
                return rgb[y0:y1, x0:x1]
        return _fallback_face_crop(rgb)

    def pose_score(self, person_rgb: np.ndarray, pred_rgb: np.ndarray) -> Tuple[float, float, str]:
        if self._pose is None:
            score = _pose_score_fallback(person_rgb, pred_rgb, None)
            return 1.0 - score, score, "edge_ncc_fallback"

        person_u8 = (np.clip(person_rgb, 0, 1) * 255).astype(np.uint8)
        pred_u8 = (np.clip(pred_rgb, 0, 1) * 255).astype(np.uint8)
        r1 = self._pose.process(person_u8)
        r2 = self._pose.process(pred_u8)
        if (not r1.pose_landmarks) or (not r2.pose_landmarks):
            score = _pose_score_fallback(person_rgb, pred_rgb, None)
            return 1.0 - score, score, "edge_ncc_fallback"

        idxs = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        p1 = r1.pose_landmarks.landmark
        p2 = r2.pose_landmarks.landmark
        pts1 = []
        pts2 = []
        for i in idxs:
            if p1[i].visibility > 0.4 and p2[i].visibility > 0.4:
                pts1.append((p1[i].x, p1[i].y))
                pts2.append((p2[i].x, p2[i].y))
        if len(pts1) < 6:
            score = _pose_score_fallback(person_rgb, pred_rgb, None)
            return 1.0 - score, score, "edge_ncc_fallback"

        pts1_a = np.asarray(pts1, dtype=np.float32)
        pts2_a = np.asarray(pts2, dtype=np.float32)
        torso = 0.5 * (
            np.linalg.norm(np.asarray([p1[11].x - p1[12].x, p1[11].y - p1[12].y]))
            + np.linalg.norm(np.asarray([p1[23].x - p1[24].x, p1[23].y - p1[24].y]))
        )
        torso = max(float(torso), 1e-3)
        err = float(np.mean(np.linalg.norm(pts1_a - pts2_a, axis=1)) / torso)
        score = float(np.exp(-err))
        return err, score, "mediapipe_pose"


def _pose_score_fallback(person_rgb: np.ndarray, pred_rgb: np.ndarray, mask: Optional[np.ndarray]) -> float:
    h = min(person_rgb.shape[0], pred_rgb.shape[0])
    w = min(person_rgb.shape[1], pred_rgb.shape[1])
    p = _resize_rgb(person_rgb, (h, w))
    q = _resize_rgb(pred_rgb, (h, w))
    gp = _to_gray(p)
    gq = _to_gray(q)

    ep = np.hypot(ndimage.sobel(gp, axis=1), ndimage.sobel(gp, axis=0))
    eq = np.hypot(ndimage.sobel(gq, axis=1), ndimage.sobel(gq, axis=0))

    if mask is not None:
        if mask.shape != (h, w):
            mask = _resize_mask(mask, (h, w))
        # posture should be evaluated mostly outside garment-change region
        roi = (1.0 - mask) > 0.5
        if np.count_nonzero(roi) > 128:
            return _masked_ncc(ep, eq, roi.astype(np.float32))
    return _ncc(ep, eq)


def _pick_pred_cols(columns: Tuple[str, ...], pred_a: str, pred_b: str) -> Tuple[str, str]:
    if pred_a and pred_b:
        if pred_a not in columns or pred_b not in columns:
            raise ValueError("Provided pred columns are not present in CSV.")
        return pred_a, pred_b
    pred_cols = [c for c in columns if c.startswith("pred_path_")]
    if len(pred_cols) < 2:
        raise ValueError("Need at least two `pred_path_*` columns in overlap CSV.")
    return pred_cols[0], pred_cols[1]


def _pick_cloth_col(columns: Tuple[str, ...], run_suffix: str, fallback_col: str) -> str:
    col = f"cloth_path_{run_suffix}"
    return col if col in columns else fallback_col


def _safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 8:
        return None
    diff = a - b
    if np.allclose(diff, 0):
        return None
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
    except Exception:
        return None


def _evaluate_side(
    pred_rgb: np.ndarray,
    cloth_rgb: np.ndarray,
    person_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    mp_helper: MediaPipeHelper,
) -> Dict[str, float]:
    cloth_pred = _fallback_cloth_crop(pred_rgb, mask)
    cloth_ref = cloth_rgb
    cloth_ref_resized = _resize_rgb(cloth_ref, cloth_pred.shape[:2])

    cloth_hist = _hist_intersection(_hist_rgb(cloth_pred), _hist_rgb(cloth_ref_resized))
    cloth_edge = _hist_intersection(
        _edge_dir_hist(_to_gray(cloth_pred)), _edge_dir_hist(_to_gray(cloth_ref_resized))
    )
    cloth_score = 0.6 * cloth_hist + 0.4 * cloth_edge

    person_face = mp_helper.face_crop(person_rgb)
    pred_face = mp_helper.face_crop(pred_rgb)
    face_ref = _resize_rgb(person_face, pred_face.shape[:2])
    face_hist = _hist_intersection(_hist_rgb(pred_face), _hist_rgb(face_ref))
    face_ncc = _ncc(_to_gray(pred_face), _to_gray(face_ref))
    face_score = 0.5 * face_hist + 0.5 * face_ncc

    pose_err, pose_score, _ = mp_helper.pose_score(person_rgb, pred_rgb)
    if math.isnan(pose_score):
        pose_score = _pose_score_fallback(person_rgb, pred_rgb, mask)
        pose_err = 1.0 - pose_score

    return {
        "cloth_hist": float(cloth_hist),
        "cloth_edge": float(cloth_edge),
        "cloth_score": float(cloth_score),
        "face_hist": float(face_hist),
        "face_ncc": float(face_ncc),
        "face_score": float(face_score),
        "pose_error": float(pose_err),
        "pose_score": float(pose_score),
    }


def _read_csv_columns(path: Path) -> Tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return tuple(reader.fieldnames or ())


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k)) for k in fieldnames})


def run(args: argparse.Namespace) -> None:
    columns = _read_csv_columns(args.overlap_csv)
    if not columns:
        raise ValueError(f"Missing header in CSV: {args.overlap_csv}")

    pred_a_col, pred_b_col = _pick_pred_cols(columns, args.pred_a_col, args.pred_b_col)
    run_a = pred_a_col.replace("pred_path_", "")
    run_b = pred_b_col.replace("pred_path_", "")

    fallback_cloth = "cloth_path" if "cloth_path" in columns else ""
    cloth_a_col = (
        args.cloth_a_col
        if args.cloth_a_col
        else _pick_cloth_col(columns, run_a, fallback_cloth) if fallback_cloth else f"cloth_path_{run_a}"
    )
    cloth_b_col = (
        args.cloth_b_col
        if args.cloth_b_col
        else _pick_cloth_col(columns, run_b, fallback_cloth) if fallback_cloth else f"cloth_path_{run_b}"
    )

    if cloth_a_col not in columns or cloth_b_col not in columns:
        raise ValueError("Could not resolve cloth path columns for both runs.")
    if args.person_col not in columns:
        raise ValueError(f"Missing person column: {args.person_col}")

    mp_helper = MediaPipeHelper(
        enable_face=bool(args.use_mediapipe_face), enable_pose=bool(args.use_mediapipe_pose)
    )

    rows = []
    try:
        with Path(args.overlap_csv).open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"Missing header in CSV: {args.overlap_csv}")

            for idx, row in enumerate(reader):
                if args.max_samples > 0 and idx >= args.max_samples:
                    break

                pair_id = str(row.get("pair_id") or idx)
                try:
                    pred_a = _load_rgb(str(row.get(pred_a_col) or ""))
                    pred_b = _load_rgb(str(row.get(pred_b_col) or ""))
                    cloth_a = _load_rgb(str(row.get(cloth_a_col) or ""))
                    cloth_b = _load_rgb(str(row.get(cloth_b_col) or ""))
                    person = _load_rgb(str(row.get(args.person_col) or ""))
                    mask = None
                    if args.mask_col in columns:
                        mask_path = str(row.get(args.mask_col) or "")
                        if mask_path:
                            mask = _load_mask(mask_path)
                except Exception:
                    continue

                m_a = _evaluate_side(pred_a, cloth_a, person, mask, mp_helper)
                m_b = _evaluate_side(pred_b, cloth_b, person, mask, mp_helper)

                rows.append(
                    {
                        "pair_id": pair_id,
                        "run_a": run_a,
                        "run_b": run_b,
                        "cloth_score_a": m_a["cloth_score"],
                        "cloth_score_b": m_b["cloth_score"],
                        "face_score_a": m_a["face_score"],
                        "face_score_b": m_b["face_score"],
                        "pose_score_a": m_a["pose_score"],
                        "pose_score_b": m_b["pose_score"],
                        "pose_error_a": m_a["pose_error"],
                        "pose_error_b": m_b["pose_error"],
                        "cloth_hist_a": m_a["cloth_hist"],
                        "cloth_hist_b": m_b["cloth_hist"],
                        "cloth_edge_a": m_a["cloth_edge"],
                        "cloth_edge_b": m_b["cloth_edge"],
                        "face_hist_a": m_a["face_hist"],
                        "face_hist_b": m_b["face_hist"],
                        "face_ncc_a": m_a["face_ncc"],
                        "face_ncc_b": m_b["face_ncc"],
                    }
                )
    finally:
        mp_helper.close()

    if not rows:
        raise RuntimeError("No valid rows evaluated.")

    def _is_good_row(r: dict[str, object]) -> bool:
        keys = (
            "cloth_score_a",
            "cloth_score_b",
            "face_score_a",
            "face_score_b",
            "pose_score_a",
            "pose_score_b",
        )
        for k in keys:
            v = r.get(k)
            if v is None:
                return False
            try:
                fv = float(v)  # type: ignore[arg-type]
            except Exception:
                return False
            if not math.isfinite(fv):
                return False
        return True

    pair_rows = [r for r in rows if _is_good_row(r)]
    if not pair_rows:
        raise RuntimeError("No finite metric rows were produced.")

    def _summary_for(metric: str) -> Dict[str, object]:
        a = np.asarray([r[f"{metric}_a"] for r in pair_rows], dtype=np.float64)
        b = np.asarray([r[f"{metric}_b"] for r in pair_rows], dtype=np.float64)
        return {
            "metric": metric,
            "n_pairs": int(a.size),
            "run_a_mean": float(np.mean(a)),
            "run_b_mean": float(np.mean(b)),
            "mean_delta_a_minus_b": float(np.mean(a - b)),
            "median_delta_a_minus_b": float(np.median(a - b)),
            "a_better_win_rate": float(np.mean(a > b)),
            "paired_wilcoxon_p": _safe_wilcoxon(a, b),
        }

    summary_rows = [
        _summary_for("cloth_score"),
        _summary_for("face_score"),
        _summary_for("pose_score"),
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_pair_path = outdir / "fidelity_per_pair.csv"
    summary_path = outdir / "fidelity_summary.csv"
    summary_json_path = outdir / "fidelity_summary.json"

    _write_csv(
        per_pair_path,
        pair_rows,
        fieldnames=[
            "pair_id",
            "run_a",
            "run_b",
            "cloth_score_a",
            "cloth_score_b",
            "face_score_a",
            "face_score_b",
            "pose_score_a",
            "pose_score_b",
            "pose_error_a",
            "pose_error_b",
            "cloth_hist_a",
            "cloth_hist_b",
            "cloth_edge_a",
            "cloth_edge_b",
            "face_hist_a",
            "face_hist_b",
            "face_ncc_a",
            "face_ncc_b",
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        fieldnames=[
            "metric",
            "n_pairs",
            "run_a_mean",
            "run_b_mean",
            "mean_delta_a_minus_b",
            "median_delta_a_minus_b",
            "a_better_win_rate",
            "paired_wilcoxon_p",
        ],
    )
    summary_json_path.write_text(
        json.dumps(
            {
                "overlap_csv": str(args.overlap_csv),
                "run_a": run_a,
                "run_b": run_b,
                "pred_a_col": pred_a_col,
                "pred_b_col": pred_b_col,
                "cloth_a_col": cloth_a_col,
                "cloth_b_col": cloth_b_col,
                "person_col": args.person_col,
                "mask_col": args.mask_col,
                "use_mediapipe_face": bool(args.use_mediapipe_face and HAS_MEDIAPIPE),
                "use_mediapipe_pose": bool(args.use_mediapipe_pose and HAS_MEDIAPIPE),
                "rows_scored": int(len(pair_rows)),
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {per_pair_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {summary_json_path}")
    if (args.use_mediapipe_face or args.use_mediapipe_pose) and not HAS_MEDIAPIPE:
        print("Note: mediapipe not installed. Used fallback face/pose logic.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overlap-csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--pred-a-col", default="")
    p.add_argument("--pred-b-col", default="")
    p.add_argument("--cloth-a-col", default="")
    p.add_argument("--cloth-b-col", default="")
    p.add_argument("--person-col", default="person_path")
    p.add_argument("--mask-col", default="mask_path")
    p.add_argument("--max-samples", type=int, default=0, help="0 means all rows.")
    p.add_argument("--use-mediapipe-face", action="store_true")
    p.add_argument("--use-mediapipe-pose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
