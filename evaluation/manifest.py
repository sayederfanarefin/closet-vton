import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Sample:
    sample_id: str
    person_path: Path
    cloth_path: Path
    gt_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    category: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def _resolve_path(base_dir: Path, p: Optional[str]) -> Optional[Path]:
    if p is None or p == "":
        return None
    path = Path(p)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_manifest(manifest_path: str) -> List[Sample]:
    """
    Manifest format (recommended): JSONL, one object per line with keys:
      - id (or sample_id)
      - person
      - cloth
      - gt (optional but needed for most metrics)
      - mask (optional; if provided enables masked metrics)
      - category (optional)

    CSV also supported with header containing: id, person, cloth, gt?, mask?, category?
    """
    path = Path(manifest_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    base_dir = path.parent
    samples: List[Sample] = []

    if path.suffix.lower() in {".jsonl", ".json"}:
        # Treat .json as either JSONL or a JSON array.
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            rows: Iterable[Dict[str, Any]] = json.loads(text)
        else:
            rows = (json.loads(line) for line in text.splitlines() if line.strip())

        for row in rows:
            sid = str(row.get("id") or row.get("sample_id") or "")
            if not sid:
                raise ValueError(f"Missing id/sample_id in manifest row: {row}")
            person = _resolve_path(base_dir, row.get("person"))
            cloth = _resolve_path(base_dir, row.get("cloth"))
            gt = _resolve_path(base_dir, row.get("gt"))
            mask = _resolve_path(base_dir, row.get("mask"))
            if person is None or cloth is None:
                raise ValueError(f"Missing person/cloth path in row id={sid}")

            samples.append(
                Sample(
                    sample_id=sid,
                    person_path=person,
                    cloth_path=cloth,
                    gt_path=gt,
                    mask_path=mask,
                    category=row.get("category"),
                    meta={k: v for k, v in row.items() if k not in {"id", "sample_id", "person", "cloth", "gt", "mask", "category"}},
                )
            )

    elif path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = str(row.get("id") or row.get("sample_id") or "").strip()
                if not sid:
                    raise ValueError(f"Missing id/sample_id in CSV row: {row}")
                person = _resolve_path(base_dir, row.get("person"))
                cloth = _resolve_path(base_dir, row.get("cloth"))
                gt = _resolve_path(base_dir, row.get("gt"))
                mask = _resolve_path(base_dir, row.get("mask"))
                if person is None or cloth is None:
                    raise ValueError(f"Missing person/cloth path in row id={sid}")

                samples.append(
                    Sample(
                        sample_id=sid,
                        person_path=person,
                        cloth_path=cloth,
                        gt_path=gt,
                        mask_path=mask,
                        category=(row.get("category") or None),
                        meta={k: v for k, v in row.items() if k not in {"id", "sample_id", "person", "cloth", "gt", "mask", "category"}},
                    )
                )
    else:
        raise ValueError("Unsupported manifest extension. Use .jsonl/.json or .csv")

    return samples
