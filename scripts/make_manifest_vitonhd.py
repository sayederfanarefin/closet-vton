#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Create an evaluation manifest from VITON-HD (zalando-hd-resized).")
    p.add_argument("--root", required=True, help="Path to the extracted zalando-hd-resized directory.")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--mode", choices=["paired", "unpaired"], default="paired")
    p.add_argument("--out", required=True, help="Output manifest path (.jsonl).")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).resolve()
    split_dir = root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    img_dir = split_dir / "image"
    cloth_dir = split_dir / "cloth"
    mask_dir = split_dir / "agnostic-mask"
    cloth_mask_dir = split_dir / "cloth-mask"
    pairs_path = root / f"{args.split}_pairs.txt"

    rows = []

    if args.mode == "paired":
        images = sorted(img_dir.glob("*.jpg"))
        for img_path in images:
            base = img_path.stem  # e.g. 05006_00
            cloth_path = cloth_dir / f"{base}.jpg"
            if not cloth_path.exists():
                # Some datasets may have missing cloths; skip.
                continue
            mask_path = mask_dir / f"{base}_mask.png"
            cloth_mask_path = cloth_mask_dir / f"{base}.jpg"
            rows.append(
                {
                    "id": base,
                    "person": str(img_path),
                    "cloth": str(cloth_path),
                    # Paired protocol: GT equals the target worn image (same as person image in VITON-style datasets).
                    "gt": str(img_path),
                    "mask": str(mask_path) if mask_path.exists() else None,
                    # Extra fields consumed by advanced metrics (e.g., PHaR garment fidelity).
                    "cloth_mask": str(cloth_mask_path) if cloth_mask_path.exists() else None,
                    "category": "upper",
                }
            )
            if args.limit and len(rows) >= args.limit:
                break

    else:
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
        with pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                person_name, cloth_name = line.split()
                person_path = img_dir / person_name
                cloth_path = cloth_dir / cloth_name
                if not person_path.exists() or not cloth_path.exists():
                    continue
                base = Path(person_name).stem
                cloth_mask_path = cloth_mask_dir / cloth_name
                mask_path = mask_dir / f"{base}_mask.png"
                rows.append(
                    {
                        # In VITON-style unpaired evaluation, each person appears once in pairs.txt.
                        # Use the person id as the sample id so baseline repos that save outputs
                        # as <person_name>.<ext> can be scored without renaming files.
                        "id": base,
                        "pair_cloth_id": Path(cloth_name).stem,
                        "person": str(person_path),
                        "cloth": str(cloth_path),
                        "gt": None,
                        "mask": str(mask_path) if mask_path.exists() else None,
                        "cloth_mask": str(cloth_mask_path) if cloth_mask_path.exists() else None,
                        "category": "upper",
                    }
                )
                if args.limit and len(rows) >= args.limit:
                    break

    out_path = Path(args.out).resolve()
    _write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
