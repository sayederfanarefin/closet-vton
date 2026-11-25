#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Pair:
    cloth_id: str
    person_id: str
    category: Optional[str]

    @property
    def pair_id(self) -> str:
        return f"{self.cloth_id}__{self.person_id}"


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise SystemExit(f"Expected JSON object at {path}:{lineno}, got {type(obj).__name__}")
            yield lineno, obj


def _resolve_results_path(run_or_results: Path) -> Path:
    p = run_or_results
    if p.is_dir():
        rp = p / "results.jsonl"
        if not rp.exists():
            raise SystemExit(f"Run dir missing results.jsonl: {p}")
        return rp
    if p.is_file():
        if p.name != "results.jsonl":
            raise SystemExit(f"Expected a run dir or results.jsonl file, got: {p}")
        return p
    raise SystemExit(f"Path not found: {p}")


def _extract_pair_ids(row: Dict[str, Any]) -> Tuple[str, str]:
    sid = str(row.get("id") or "").strip()
    if "__" in sid:
        cloth_id, person_id = sid.split("__", 1)
        return cloth_id, person_id
    cloth_path = str(row.get("cloth_path") or "")
    person_path = str(row.get("person_path") or "")
    if cloth_path and person_path:
        return Path(cloth_path).stem, Path(person_path).stem
    raise ValueError(f"Cannot extract pair ids from row (missing id/paths): {row}")


def _relpath(p: Path, base_dir: Path, *, relative_paths: bool) -> str:
    if not relative_paths:
        return str(p)
    return os.path.relpath(str(p), start=str(base_dir))


def _ensure_symlink(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        return
    # Create a relative symlink when possible (more portable).
    try:
        rel = os.path.relpath(str(src), start=str(dst.parent))
        os.symlink(rel, str(dst))
    except Exception:
        os.symlink(str(src), str(dst))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare a reproducible benchmark (pairs.txt + manifest.jsonl + symlinked dataroot) for running external "
            "VTON baselines on the *same pair list* processed by an existing run (e.g., mytest3curt).\n\n"
            "Why: runs like mytest3curt may have used a huge cartesian manifest but only processed a subset. "
            "We extract the actual (cloth_id, person_id) pairs from results.jsonl and build a compact benchmark "
            "that external repos can consume (VITON-HD style pairs files)."
        )
    )
    p.add_argument(
        "--source-run",
        default="runs/20260205_184710_7968b6_mytest3curt",
        help="Run dir (contains results.jsonl) or a direct path to results.jsonl.",
    )
    p.add_argument(
        "--dataset-root",
        default="data/viton-hd/zalando-hd-resized",
        help="Path to VITON-HD 'zalando-hd-resized' root (contains test/ and test_pairs.txt).",
    )
    p.add_argument(
        "--cloth-source",
        choices=["cloth", "image"],
        default="cloth",
        help=(
            "Where to read the cloth reference from: "
            "'cloth' uses VITON-HD test/cloth (proper garment), "
            "'image' uses VITON-HD test/image (person-as-cloth; matches mytest3curt)."
        ),
    )
    p.add_argument(
        "--cloth-mask-mode",
        choices=["auto", "dataset", "ones"],
        default="auto",
        help=(
            "How to populate test/cloth-mask in the generated dataroot: "
            "'dataset' symlinks VITON-HD test/cloth-mask, "
            "'ones' generates all-white masks per cloth id, "
            "'auto' picks 'ones' when --cloth-source=image else 'dataset'."
        ),
    )
    p.add_argument(
        "--white-mask-size",
        default="1024x768",
        help="Mask size for --cloth-mask-mode=ones, formatted as HxW (default: 1024x768).",
    )
    p.add_argument(
        "--out-root",
        default="external_vtons/bench_mytest3curt_ok",
        help="Output directory for the benchmark bundle (default: external_vtons/bench_mytest3curt_ok).",
    )
    p.add_argument(
        "--statuses",
        default="ok",
        help="Comma-separated statuses to include from results.jsonl (default: ok).",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="If >0, cap the number of extracted pairs (after filtering).",
    )
    p.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write manifest paths relative to the output manifest directory (recommended).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files (pairs + manifest). Symlinks are left as-is.",
    )
    return p.parse_args()


def _parse_hw(s: str) -> Tuple[int, int]:
    s = s.strip().lower().replace(" ", "")
    if "x" not in s:
        raise ValueError("Expected format HxW, e.g. 1024x768")
    h_str, w_str = s.split("x", 1)
    h = int(h_str)
    w = int(w_str)
    if h <= 0 or w <= 0:
        raise ValueError("H and W must be positive")
    return h, w


def _ensure_white_cloth_masks(
    *,
    out_dir: Path,
    cloth_ids: Sequence[str],
    size_hw: Tuple[int, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    white_path = out_dir / "__ALL_WHITE__.jpg"
    if not white_path.exists():
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise SystemExit("Pillow is required for --cloth-mask-mode=ones (pip install Pillow).") from exc

        h, w = size_hw
        img = Image.new("L", (w, h), color=255)
        img.save(white_path, format="JPEG", quality=95)

    for cid in cloth_ids:
        dst = out_dir / f"{cid}.jpg"
        _ensure_symlink(dst, white_path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    source_run = Path(args.source_run).expanduser()
    if not source_run.is_absolute():
        source_run = (repo_root / source_run).resolve()
    results_path = _resolve_results_path(source_run)

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (repo_root / dataset_root).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    test_dir = dataset_root / "test"
    if not test_dir.exists():
        raise SystemExit(f"Dataset test/ not found under: {dataset_root}")

    img_dir = test_dir / "image"
    cloth_dir = test_dir / "cloth"
    mask_dir = test_dir / "agnostic-mask"
    cloth_mask_dir = test_dir / "cloth-mask"

    for req in (img_dir, cloth_dir, mask_dir, cloth_mask_dir):
        if not req.exists():
            raise SystemExit(f"Missing required VITON-HD folder: {req}")

    cloth_source = str(args.cloth_source).strip().lower()
    if cloth_source not in {"cloth", "image"}:
        raise SystemExit("--cloth-source must be one of: cloth,image")
    cloth_img_dir = cloth_dir if cloth_source == "cloth" else img_dir

    cloth_mask_mode = str(args.cloth_mask_mode).strip().lower()
    if cloth_mask_mode == "auto":
        cloth_mask_mode = "ones" if cloth_source == "image" else "dataset"
    if cloth_mask_mode not in {"dataset", "ones"}:
        raise SystemExit("--cloth-mask-mode must be one of: auto,dataset,ones")
    white_mask_size_hw = _parse_hw(str(args.white_mask_size))

    out_root = Path(args.out_root).expanduser()
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    statuses = {s.strip() for s in str(args.statuses).split(",") if s.strip()}
    if not statuses:
        raise SystemExit("--statuses must not be empty")

    # Extract unique pairs from results.jsonl.
    pairs: List[Pair] = []
    seen: Set[Tuple[str, str]] = set()
    skipped_missing = 0
    for lineno, row in _iter_jsonl(results_path):
        status = str(row.get("status") or "").strip()
        if status not in statuses:
            continue
        try:
            cloth_id, person_id = _extract_pair_ids(row)
        except Exception as exc:
            raise SystemExit(f"Failed extracting ids at {results_path}:{lineno}: {exc}") from exc

        key = (cloth_id, person_id)
        if key in seen:
            continue
        seen.add(key)

        person_path = img_dir / f"{person_id}.jpg"
        cloth_path = cloth_img_dir / f"{cloth_id}.jpg"
        person_mask = mask_dir / f"{person_id}_mask.png"
        cloth_mask = cloth_mask_dir / f"{cloth_id}.jpg"

        if not (person_path.exists() and cloth_path.exists()):
            skipped_missing += 1
            continue

        pairs.append(Pair(cloth_id=cloth_id, person_id=person_id, category=row.get("category")))
        if args.max_pairs and len(pairs) >= args.max_pairs:
            break

    print(f"[prepare_external_vton_benchmark] results={results_path}")
    print(f"[prepare_external_vton_benchmark] dataset_root={dataset_root}")
    print(f"[prepare_external_vton_benchmark] cloth_source={cloth_source}")
    print(f"[prepare_external_vton_benchmark] cloth_mask_mode={cloth_mask_mode}")
    print(f"[prepare_external_vton_benchmark] statuses={sorted(statuses)}")
    print(f"[prepare_external_vton_benchmark] pairs={len(pairs)} (skipped_missing={skipped_missing})")
    print(f"[prepare_external_vton_benchmark] out_root={out_root}")

    # Write VITON-style pairs file (person_filename cloth_filename)
    pairs_lines = [f"{p.person_id}.jpg {p.cloth_id}.jpg" for p in pairs]
    pairs_text = "\n".join(pairs_lines) + "\n" if pairs_lines else ""

    pairs_txt = out_root / "pairs.txt"
    test_pairs_txt = out_root / "test_pairs.txt"
    test_pairs_unpaired_txt = out_root / "test_pairs_unpaired.txt"
    manifest_jsonl = out_root / "manifest.jsonl"
    meta_json = out_root / "meta.json"

    if args.force or (not pairs_txt.exists()):
        _write_text(pairs_txt, pairs_text)
    if args.force or (not test_pairs_txt.exists()):
        _write_text(test_pairs_txt, pairs_text)
    if args.force or (not test_pairs_unpaired_txt.exists()):
        _write_text(test_pairs_unpaired_txt, pairs_text)

    # Write manifest suitable for evaluation.score_folder (and for our internal comparisons).
    out_base_dir = manifest_jsonl.parent
    if args.force or (not manifest_jsonl.exists()):
        with manifest_jsonl.open("w", encoding="utf-8") as f:
            for p in pairs:
                person_path = img_dir / f"{p.person_id}.jpg"
                cloth_path = cloth_img_dir / f"{p.cloth_id}.jpg"
                person_mask = mask_dir / f"{p.person_id}_mask.png"
                if cloth_mask_mode == "ones":
                    cloth_mask = out_root / "dataroot" / "test" / "cloth-mask" / f"{p.cloth_id}.jpg"
                else:
                    cloth_mask = cloth_mask_dir / f"{p.cloth_id}.jpg"
                row = {
                    "id": p.pair_id,
                    "person": _relpath(person_path, out_base_dir, relative_paths=args.relative_paths),
                    "cloth": _relpath(cloth_path, out_base_dir, relative_paths=args.relative_paths),
                    "gt": None,
                    "mask": _relpath(person_mask, out_base_dir, relative_paths=args.relative_paths)
                    if person_mask.exists()
                    else None,
                    "cloth_mask": (
                        _relpath(cloth_mask, out_base_dir, relative_paths=args.relative_paths)
                        if (cloth_mask_mode == "ones" or cloth_mask.exists())
                        else None
                    ),
                    "category": p.category or "upper",
                    "pair_cloth_id": p.cloth_id,
                    "pair_person_id": p.person_id,
                }
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    # Create a symlinked dataroot so baseline repos can read our custom pairs without modifying the original dataset.
    dataroot = out_root / "dataroot"
    dataroot_test = dataroot / "test"
    dataroot.mkdir(parents=True, exist_ok=True)
    dataroot_test.mkdir(parents=True, exist_ok=True)

    # Copy pairs files expected by common baseline repos.
    # - Many use <dataroot>/test_pairs.txt
    # - Some (e.g., CatVTON) uses <dataroot>/test_pairs_unpaired.txt
    for fname in ("test_pairs.txt", "test_pairs_unpaired.txt", "pairs.txt"):
        src = out_root / fname
        dst = dataroot / fname
        if src.exists() and (args.force or (not dst.exists())):
            _write_text(dst, src.read_text(encoding="utf-8"))

    # Symlink required subfolders.
    subdirs = [
        "image",
        "agnostic-mask",
        "agnostic-v3.2",
        "image-densepose",
        "image-parse-v3",
        "image-parse-agnostic-v3.2",
        "openpose_img",
        "openpose_json",
    ]
    for sd in subdirs:
        src = test_dir / sd
        dst = dataroot_test / sd
        if src.exists():
            _ensure_symlink(dst, src)

    # Cloth images in dataroot/test/cloth
    _ensure_symlink(dataroot_test / "cloth", cloth_img_dir)

    # Cloth masks in dataroot/test/cloth-mask
    if cloth_mask_mode == "dataset":
        _ensure_symlink(dataroot_test / "cloth-mask", cloth_mask_dir)
    else:
        # Some baselines (e.g. HR-VITON) always load both "paired" and "unpaired" cloths,
        # which means they may try to load cloth-mask for filenames from *both* columns
        # of the pairs file. In person-as-cloth mode those "paired cloth" names are the
        # person ids, so generate masks for both ids.
        cloth_ids = sorted({p.cloth_id for p in pairs} | {p.person_id for p in pairs})
        _ensure_white_cloth_masks(
            out_dir=dataroot_test / "cloth-mask",
            cloth_ids=cloth_ids,
            size_hw=white_mask_size_hw,
        )

    _write_json(
        meta_json,
        {
            "source_results": str(results_path),
            "dataset_root": str(dataset_root),
            "cloth_source": cloth_source,
            "cloth_mask_mode": cloth_mask_mode,
            "statuses": sorted(statuses),
            "pairs": len(pairs),
            "skipped_missing": skipped_missing,
            "out_root": str(out_root),
            "dataroot": str(dataroot),
            "files": {
                "pairs_txt": str(pairs_txt),
                "manifest_jsonl": str(manifest_jsonl),
                "dataroot_test_pairs": str(dataroot / "test_pairs.txt"),
            },
        },
    )

    print(f"[prepare_external_vton_benchmark] wrote: {pairs_txt}")
    print(f"[prepare_external_vton_benchmark] wrote: {manifest_jsonl}")
    print(f"[prepare_external_vton_benchmark] dataroot: {dataroot}")


if __name__ == "__main__":
    main()
