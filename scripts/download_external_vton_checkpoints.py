#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import requests
from tqdm import tqdm


HRVITON_CHECKPOINTS: Dict[str, Tuple[str, str]] = {
    # name -> (filename, google_drive_file_id)
    "tocg": ("mtviton.pth", "1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ"),
    "gen": ("gen.pth", "1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy"),
}

STABLEVITON_SHARE_LINK = (
    "https://kaistackr-my.sharepoint.com/:f:/g/personal/"
    "rlawjdghek_kaist_ac_kr/EjzAZHJu9MlEoKIxG4tqPr0BM_Ry20NHyNw5Sic2vItxiA"
)
STABLEVITON_DEFAULT_FILE = "VITONHD.ckpt"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_stream(
    *,
    response: requests.Response,
    dst: Path,
    desc: str,
    chunk_size: int = 1024 * 1024,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = response.headers.get("content-length")
    total_i = int(total) if (total and total.isdigit()) else 0

    if dst.exists() and total_i > 0 and dst.stat().st_size == total_i:
        print(f"[skip] exists: {dst} ({total_i} bytes)")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink(missing_ok=True)

    with tmp.open("wb") as f, tqdm(
        total=total_i if total_i > 0 else None,
        unit="B",
        unit_scale=True,
        desc=desc,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            bar.update(len(chunk))
    tmp.replace(dst)


def _extract_google_drive_form(html: str) -> Tuple[str, Dict[str, str]]:
    # We expect a "virus scan warning" page with a download form.
    m = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html)
    if not m:
        raise RuntimeError("Could not find google drive download form.")
    action = m.group(1)
    inputs = re.findall(r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]*value="([^"]*)"', html)
    params = {k: v for k, v in inputs}
    if not params.get("id"):
        raise RuntimeError("Google drive form missing file id.")
    return action, params


def download_google_drive(*, file_id: str, dst: Path, label: str, dry_run: bool) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] exists: {dst}")
        return
    if dry_run:
        print(f"[dry-run] google-drive id={file_id} -> {dst}")
        return

    s = requests.Session()
    # Step 1: request the download page (often returns a warning HTML)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = s.get(url, timeout=60)
    r.raise_for_status()

    # Step 2: follow the form to the real download endpoint
    action, params = _extract_google_drive_form(r.text)
    r2 = s.get(action, params=params, stream=True, timeout=60, allow_redirects=True)
    r2.raise_for_status()

    ct = (r2.headers.get("content-type") or "").lower()
    if "text/html" in ct:
        raise RuntimeError("Google Drive returned HTML instead of a file (quota/auth issue).")

    _download_stream(response=r2, dst=dst, desc=label)


def _stableviton_list_sharepoint_files(session: requests.Session) -> Tuple[str, str, Iterable[Dict[str, str]]]:
    """
    Redeem the share link to get cookies, then discover the folder root via redirect
    and list files via SharePoint REST API.

    Returns: (base_url, site_prefix, files)
    """
    start = STABLEVITON_SHARE_LINK + "?download=1"
    r = session.get(start, timeout=60, allow_redirects=True)
    r.raise_for_status()
    final = r.url
    u = urlparse(final)
    qs = parse_qs(u.query)
    root = (qs.get("RootFolder") or [""])[0]
    root = unquote(root)
    if not root.startswith("/"):
        raise RuntimeError(f"Could not discover RootFolder from redirect URL: {final}")

    parts = [p for p in root.strip("/").split("/") if p]
    if len(parts) < 2:
        raise RuntimeError(f"Unexpected RootFolder: {root}")
    site_prefix = "/" + "/".join(parts[:2])  # /personal/<user>
    base_url = f"{u.scheme}://{u.netloc}"

    api = (
        base_url
        + site_prefix
        + "/_api/web/GetFolderByServerRelativeUrl("
        + requests.utils.quote(f"'{root}'", safe="")
        + ")/Files?$select=Name,ServerRelativeUrl,Length"
    )
    resp = session.get(api, headers={"Accept": "application/json;odata=nometadata"}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    files = data.get("value") or []
    if not isinstance(files, list):
        raise RuntimeError("Unexpected SharePoint API response.")
    return base_url, site_prefix, files


def download_stableviton(*, dst_dir: Path, file_name: str, dry_run: bool) -> Path:
    dst = dst_dir / file_name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] exists: {dst}")
        return dst
    if dry_run:
        print(f"[dry-run] stableviton {file_name} -> {dst}")
        return dst

    s = requests.Session()
    base_url, _site_prefix, files = _stableviton_list_sharepoint_files(s)
    hit = None
    for f in files:
        if str(f.get("Name") or "") == file_name:
            hit = f
            break
    if hit is None:
        names = [str(f.get("Name") or "") for f in files]
        raise RuntimeError(f"StableVITON file not found in share folder: {file_name}. Available: {names}")

    rel = str(hit.get("ServerRelativeUrl") or "")
    if not rel.startswith("/"):
        raise RuntimeError("SharePoint file missing ServerRelativeUrl.")
    url = base_url + rel + "?download=1"
    r = s.get(url, stream=True, timeout=60)
    r.raise_for_status()
    _download_stream(response=r, dst=dst, desc=f"stableviton:{file_name}")
    return dst


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download external VTON baseline checkpoints (HR-VITON + StableVITON).")
    p.add_argument(
        "--out-root",
        default="external_vtons/checkpoints",
        help="Where to write checkpoints (default: external_vtons/checkpoints).",
    )
    p.add_argument("--skip-hrviton", action="store_true", help="Skip HR-VITON checkpoints.")
    p.add_argument("--skip-stableviton", action="store_true", help="Skip StableVITON checkpoints.")
    p.add_argument(
        "--stableviton-file",
        default=STABLEVITON_DEFAULT_FILE,
        help=f"StableVITON checkpoint filename to download (default: {STABLEVITON_DEFAULT_FILE}).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_root = Path(args.out_root).expanduser()
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    _ensure_dir(out_root)

    hrviton_dir = out_root / "hrviton"
    stableviton_dir = out_root / "stableviton"

    results: Dict[str, str] = {}

    if not args.skip_hrviton:
        _ensure_dir(hrviton_dir)
        for key, (fname, file_id) in HRVITON_CHECKPOINTS.items():
            dst = hrviton_dir / fname
            download_google_drive(file_id=file_id, dst=dst, label=f"hrviton:{fname}", dry_run=bool(args.dry_run))
            results[f"hrviton_{key}"] = str(dst)

    if not args.skip_stableviton:
        _ensure_dir(stableviton_dir)
        dst = download_stableviton(
            dst_dir=stableviton_dir,
            file_name=str(args.stableviton_file),
            dry_run=bool(args.dry_run),
        )
        results["stableviton_ckpt"] = str(dst)

    print("\nDownloaded paths:")
    print(json.dumps(results, indent=2, sort_keys=True))

    if not args.dry_run:
        print("\nSuggested run command:")
        print(
            "python3 scripts/run_external_vton_both_cloth_modes.py --score "
            f"--hrviton-tocg {hrviton_dir / HRVITON_CHECKPOINTS['tocg'][0]} "
            f"--hrviton-gen {hrviton_dir / HRVITON_CHECKPOINTS['gen'][0]} "
            f"--stableviton-ckpt {stableviton_dir / str(args.stableviton_file)}"
        )


if __name__ == "__main__":
    main()
