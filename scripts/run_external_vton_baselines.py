#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Baseline:
    key: str
    repo_url: str
    repo_dir: Path
    conda_env: str


IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, check=True)


def _ensure_repo(repo_dir: Path, repo_url: str, *, dry_run: bool) -> None:
    if repo_dir.exists() and (repo_dir / ".git").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", repo_url, str(repo_dir)]
    if dry_run:
        print("+", " ".join(cmd))
        return
    _run(cmd)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _patch_once(path: Path, *, marker: str, transform) -> bool:
    """
    Apply a patch to `path` once. Returns True if modified.
    `marker` is checked as a simple substring.
    """
    src = _read_text(path)
    if marker in src:
        return False
    dst = transform(src)
    if dst == src:
        raise RuntimeError(f"Patch transform made no changes for {path}")
    _write_text(path, dst)
    return True


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _count_images(path: Path, *, recursive: bool) -> int:
    if not path.exists():
        return 0
    it = path.rglob("*") if recursive else path.iterdir()
    n = 0
    for p in it:
        try:
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                n += 1
        except OSError:
            continue
    return n


def _count_images_fast(path: Path, *, recursive: bool) -> int:
    if not path.exists():
        return 0
    exts = IMAGE_EXTS
    if not recursive:
        n = 0
        try:
            with os.scandir(path) as it:
                for entry in it:
                    try:
                        if entry.is_file() and entry.name.lower().endswith(exts):
                            n += 1
                    except OSError:
                        continue
        except OSError:
            return 0
        return n

    n = 0
    stack: List[Path] = [path]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_file():
                            if entry.name.lower().endswith(exts):
                                n += 1
                        elif entry.is_dir():
                            stack.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    return n


def _format_duration_s(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    if seconds < 0:
        return "?"
    seconds_i = int(seconds + 0.5)
    m, s = divmod(seconds_i, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return f"{d}d{h:02}h{m:02}m"
    if h > 0:
        return f"{h:02}h{m:02}m{s:02}s"
    if m > 0:
        return f"{m:02}m{s:02}s"
    return f"{s}s"


def _start_parent_progress_monitor(
    *,
    enabled: bool,
    baseline_key: str,
    watch_dir: Path,
    expected: int,
    recursive: bool,
    interval_s: float,
) -> Optional[Tuple[threading.Event, threading.Thread]]:
    if (not enabled) or expected <= 0:
        return None
    stop = threading.Event()

    def _render_line(done: int, elapsed_s: float) -> str:
        done = min(done, expected)
        pct = (done / expected) if expected else 0.0
        width = 24
        filled = min(width, max(0, int(round(pct * width))))
        bar = "#" * filled + "-" * (width - filled)
        rate = (done / elapsed_s) if elapsed_s > 0 else 0.0
        eta = ((expected - done) / rate) if rate > 0 else None
        return (
            f"[progress][{baseline_key}] {done}/{expected} ({pct:.1%}) "
            f"[{bar}] rate={rate:.3f} img/s eta={_format_duration_s(eta)}"
        )

    def _worker() -> None:
        start = time.time()
        done0 = _count_images_fast(watch_dir, recursive=recursive)
        print(_render_line(done0, max(1e-6, time.time() - start)), flush=True)
        while not stop.wait(max(1.0, float(interval_s))):
            done = _count_images_fast(watch_dir, recursive=recursive)
            elapsed = time.time() - start
            print(_render_line(done, elapsed), flush=True)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return stop, t


def _stop_parent_progress_monitor(mon: Optional[Tuple[threading.Event, threading.Thread]]) -> None:
    if mon is None:
        return
    stop, t = mon
    stop.set()
    t.join(timeout=5)


def _find_existing_scored_run(
    runs_outdir: Path,
    *,
    run_name: str,
    manifest_sha256: str,
    pred_dir: Path,
    expected_samples: int,
) -> Optional[Path]:
    pred_dir_resolved = pred_dir.resolve()
    if not runs_outdir.exists():
        return None
    for d in sorted(runs_outdir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        summary_path = d / "summary.json"
        if not meta_path.exists() or not summary_path.exists():
            continue
        try:
            meta = json.loads(_read_text(meta_path))
            summary = json.loads(_read_text(summary_path))
        except Exception:
            continue
        if str(meta.get("run_name") or "") != run_name:
            continue
        if str(meta.get("manifest_sha256") or "") != manifest_sha256:
            continue
        try:
            meta_pred = Path(str(meta.get("pred_dir") or "")).resolve()
        except Exception:
            continue
        if meta_pred != pred_dir_resolved:
            continue
        try:
            samples_total = int(summary.get("samples_total"))
        except Exception:
            continue
        if samples_total != int(expected_samples):
            continue
        return d
    return None


@dataclass(frozen=True)
class _GpuMem:
    used_mib: int
    free_mib: int
    total_mib: int


def _try_query_gpu_mem() -> Optional[_GpuMem]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) != 3:
        return None
    try:
        used, free, total = (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None
    return _GpuMem(used_mib=used, free_mib=free, total_mib=total)


def _try_query_gpu_processes(limit: int = 8) -> List[str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return []
    rows: List[Tuple[int, str, int]] = []
    for raw in (proc.stdout or "").strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            name = parts[1]
            used = int(parts[2])
        except Exception:
            continue
        rows.append((pid, name, used))
    rows.sort(key=lambda x: x[2], reverse=True)
    out: List[str] = []
    for pid, name, used in rows[: max(0, int(limit))]:
        out.append(f"pid={pid} used={used}MiB name={name}")
    return out


def _has_hf_token() -> bool:
    if os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"):
        return True
    candidates = [
        Path.home() / ".huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "token",
    ]
    return any(p.exists() for p in candidates)


def _patch_ladi(repo_dir: Path) -> None:
    """
    LaDI-VTON's default inference saves as <im_name>.jpg, which overwrites when the same person appears multiple times.
    Patch it to save as <cloth_id>__<person_id>.jpg instead.
    """
    path = repo_dir / "src" / "inference.py"
    marker = "codex_multi_pairs_filename_patch_v1"

    def transform(src: str) -> str:
        s = src
        # Ensure c_name is in outputlist so batch has both ids.
        s = s.replace(
            "outputlist = ['image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name', 'cloth']",
            "outputlist = ['image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name', 'c_name', 'cloth']",
        )
        # Adjust zip to include c_name and create unique filename.
        s = s.replace(
            'for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):',
            'for gen_image, cat, im_name, c_name in zip(generated_images, category, batch["im_name"], batch["c_name"]):',
        )
        # Insert a filename rewrite just before use_png logic.
        needle = "if args.use_png:"
        if needle not in s:
            raise RuntimeError("Could not find expected save block in LaDI-VTON inference.py")
        s = s.replace(
            needle,
            "# codex_multi_pairs_filename_patch_v1\n"
            "            name = f\"{Path(c_name).stem}__{Path(im_name).stem}.jpg\"\n"
            f"            {needle}",
        )
        return s

    changed = _patch_once(path, marker=marker, transform=transform)
    if changed:
        print(f"[patch] LaDI-VTON: patched {path}")

    # LaDI-VTON uses torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', ...) which
    # pulls a separate cached copy of hubconf.py. That hubconf hard-codes the base SD2-inpainting id,
    # so even if the user passes --pretrained_model_name_or_path it may still try to download the gated model.
    # Patch inference to use the *local* repo's hubconf and plumb through the base model id.
    marker_hub = "codex_local_torch_hub_patch_v1"

    def transform_hub(src: str) -> str:
        s = src
        old = (
            "    # Load the trained models from the hub\n"
            "    unet = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='extended_unet',\n"
            "                          dataset=args.dataset)\n"
            "    emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='emasc', dataset=args.dataset)\n"
            "    inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='inversion_adapter',\n"
            "                                       dataset=args.dataset)\n"
            "    tps, refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module',\n"
            "                                     dataset=args.dataset)\n"
        )
        if old not in s:
            raise RuntimeError("Could not find expected torch.hub.load block in LaDI-VTON inference.py")
        new = (
            "    # Load the trained models from the local repo hubconf (avoids downloading a separate torch.hub copy).\n"
            f"    # {marker_hub}\n"
            "    hub_dir = str(PROJECT_ROOT)\n"
            "    unet = torch.hub.load(\n"
            "        repo_or_dir=hub_dir,\n"
            "        source='local',\n"
            "        model='extended_unet',\n"
            "        dataset=args.dataset,\n"
            "        pretrained_model_name_or_path=args.pretrained_model_name_or_path,\n"
            "    )\n"
            "    emasc = torch.hub.load(repo_or_dir=hub_dir, source='local', model='emasc', dataset=args.dataset)\n"
            "    inversion_adapter = torch.hub.load(\n"
            "        repo_or_dir=hub_dir,\n"
            "        source='local',\n"
            "        model='inversion_adapter',\n"
            "        dataset=args.dataset,\n"
            "        pretrained_model_name_or_path=args.pretrained_model_name_or_path,\n"
            "    )\n"
            "    tps, refinement = torch.hub.load(repo_or_dir=hub_dir, source='local', model='warping_module', dataset=args.dataset)\n"
        )
        return s.replace(old, new, 1)

    changed = _patch_once(path, marker=marker_hub, transform=transform_hub)
    if changed:
        print(f"[patch] LaDI-VTON: patched torch.hub usage in {path}")

    hubconf_path = repo_dir / "hubconf.py"
    marker_hubconf = "codex_pretrained_model_arg_patch_v1"

    def transform_hubconf(src: str) -> str:
        s = src
        s = s.replace(
            "def inversion_adapter(dataset: Literal['dresscode', 'vitonhd']):",
            "def inversion_adapter(dataset: Literal['dresscode', 'vitonhd'], pretrained_model_name_or_path: str = \"stabilityai/stable-diffusion-2-inpainting\"):",
        )
        s = s.replace(
            'UNet2DConditionModel.load_config("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")',
            'UNet2DConditionModel.load_config(pretrained_model_name_or_path, subfolder="text_encoder")',
        )
        s = s.replace(
            "def extended_unet(dataset: Literal['dresscode', 'vitonhd']):",
            "def extended_unet(dataset: Literal['dresscode', 'vitonhd'], pretrained_model_name_or_path: str = \"stabilityai/stable-diffusion-2-inpainting\"):",
        )
        s = s.replace(
            'UNet2DConditionModel.load_config("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")',
            'UNet2DConditionModel.load_config(pretrained_model_name_or_path, subfolder="unet")',
        )
        if marker_hubconf not in s:
            s = s.replace(
                "dependencies = ['torch', 'diffusers', 'transformers']",
                f"dependencies = ['torch', 'diffusers', 'transformers']\n# {marker_hubconf}",
                1,
            )
        return s

    changed = _patch_once(hubconf_path, marker=marker_hubconf, transform=transform_hubconf)
    if changed:
        print(f"[patch] LaDI-VTON: patched hubconf base model arg in {hubconf_path}")


def _patch_idm(repo_dir: Path) -> None:
    """
    IDM-VTON's default inference saves as <im_name>, which overwrites for repeated persons.
    Patch it to save as <cloth_id>__<person_id>.png.
    """
    path = repo_dir / "inference.py"
    marker = "codex_multi_pairs_filename_patch_v1"

    def transform(src: str) -> str:
        s = src
        target = "torchvision.utils.save_image(x_sample,os.path.join(args.output_dir,sample['im_name'][i]))"
        if target not in s:
            raise RuntimeError("Could not find expected save_image line in IDM-VTON inference.py")
        repl = (
            "# codex_multi_pairs_filename_patch_v1\n"
            "                        im_stem = os.path.splitext(sample['im_name'][i])[0]\n"
            "                        c_stem = os.path.splitext(sample['c_name'][i])[0]\n"
            "                        out_name = f\"{c_stem}__{im_stem}.png\"\n"
            "                        torchvision.utils.save_image(x_sample, os.path.join(args.output_dir, out_name))"
        )
        s = s.replace(target, repl)
        return s

    changed = _patch_once(path, marker=marker, transform=transform)
    if changed:
        print(f"[patch] IDM-VTON: patched {path}")

    marker_progress = "codex_progress_bar_patch_v1"

    def transform_progress(src: str) -> str:
        s = src
        if marker_progress not in s:
            needle_logger = "logger = get_logger(__name__, log_level=\"INFO\")"
            if needle_logger in s:
                s = s.replace(
                    needle_logger,
                    needle_logger
                    + "\n\n"
                    f"# {marker_progress}\n"
                    "try:\n"
                    "    from tqdm import tqdm  # type: ignore\n"
                    "except Exception:\n"
                    "    tqdm = None\n",
                    1,
                )
            else:
                # Fallback: put it near the top of the file.
                s = s.replace(
                    "import argparse\n",
                    "import argparse\n\n"
                    f"# {marker_progress}\n"
                    "try:\n"
                    "    from tqdm import tqdm  # type: ignore\n"
                    "except Exception:\n"
                    "    tqdm = None\n\n",
                    1,
                )

        loop = "                for sample in test_dataloader:\n"
        if loop not in s:
            raise RuntimeError("Could not find expected test_dataloader loop in IDM-VTON inference.py")
        if "iterator = tqdm(test_dataloader" not in s:
            s = s.replace(
                loop,
                "                iterator = (\n"
                "                    tqdm(test_dataloader, total=len(test_dataloader), unit='batch') if tqdm is not None else test_dataloader\n"
                "                )\n"
                "                for sample in iterator:\n",
                1,
            )
        return s

    changed = _patch_once(path, marker=marker_progress, transform=transform_progress)
    if changed:
        print(f"[patch] IDM-VTON: added progress/ETA to {path}")

    marker_skip = "codex_skip_existing_outputs_patch_v1"

    def transform_skip(src: str) -> str:
        s = src
        needle = "                for sample in iterator:\n"
        if needle not in s:
            raise RuntimeError("Could not find expected iterator loop in IDM-VTON inference.py")
        insert = (
            "                for sample in iterator:\n"
            "                    # codex_skip_existing_outputs_patch_v1\n"
            "                    try:\n"
            "                        _out_names = []\n"
            "                        for _i in range(len(sample.get('im_name', []))):\n"
            "                            _im_stem = os.path.splitext(sample['im_name'][_i])[0]\n"
            "                            _c_stem = os.path.splitext(sample['c_name'][_i])[0]\n"
            "                            _out_names.append(f\"{_c_stem}__{_im_stem}.png\")\n"
            "                        if _out_names and all(os.path.exists(os.path.join(args.output_dir, n)) for n in _out_names):\n"
            "                            continue\n"
            "                    except Exception:\n"
            "                        pass\n"
        )
        return s.replace(needle, insert, 1)

    changed = _patch_once(path, marker=marker_skip, transform=transform_skip)
    if changed:
        print(f"[patch] IDM-VTON: enabled resume/skip-existing in {path}")


def _patch_catvton(repo_dir: Path) -> None:
    """
    CatVTON's default VITON-HD inference assumes each person appears once and saves as <person_img>.
    Patch it to save as <cloth_id>__<person_id>.png and disable the 'skip existing output' optimization.
    """
    path = repo_dir / "inference.py"
    marker = "codex_multi_pairs_filename_patch_v1"

    def transform(src: str) -> str:
        s = src
        if "from pathlib import Path" not in s:
            s = s.replace("import os", "import os\nfrom pathlib import Path")

        # Replace the block that skips already-generated outputs and uses person_img as filename.
        # Keep it conservative: only patch the specific VITONHDTestDataset.load_data snippet.
        pat = re.compile(
            r"(?s)for line in lines:\n"
            r"\s+person_img, cloth_img = line\.strip\(\)\.split\(\" \"\)\n"
            r"\s+if os\.path\.exists\(os\.path\.join\(output_dir, person_img\)\):\n"
            r"\s+\s+continue\n"
            r"\s+if self\.args\.eval_pair:\n"
            r"\s+\s+cloth_img = person_img\n"
            r"\s+data\.append\(\{\n"
            r"\s+\s+'person_name': person_img,\n"
        )
        m = pat.search(s)
        if not m:
            raise RuntimeError("Could not find expected VITONHDTestDataset loop in CatVTON inference.py")
        replacement = (
            "for line in lines:\n"
            "            person_img, cloth_img = line.strip().split(\" \")\n"
            "            if self.args.eval_pair:\n"
            "                cloth_img = person_img\n"
            "            pair_name = f\"{Path(cloth_img).stem}__{Path(person_img).stem}.png\"\n"
            "            data.append({\n"
            "                'person_name': pair_name,\n"
        )
        s = s[: m.start()] + replacement + s[m.end() :]
        # Insert marker once near top.
        s = s.replace("class InferenceDataset", f"# {marker}\n\nclass InferenceDataset", 1)
        return s

    changed = _patch_once(path, marker=marker, transform=transform)
    if changed:
        print(f"[patch] CatVTON: patched {path}")

    # Resume support: keep CatVTON's "skip if output exists" optimization, but adapt it to our unique pair filenames.
    marker_resume = "codex_skip_existing_outputs_patch_v1"

    def transform_resume(src: str) -> str:
        s = src
        old = "output_dir = os.path.join(self.args.output_dir, \"vitonhd\", 'unpaired' if not self.args.eval_pair else 'paired')\n"
        if old in s:
            # Make this match main(): <out>/<dataset_name>-<H>/<paired|unpaired>/
            s = s.replace(
                old,
                "output_dir = os.path.join(self.args.output_dir, f\"{self.args.dataset_name}-{self.args.height}\", 'unpaired' if not self.args.eval_pair else 'paired')\n",
                1,
            )
        needle = "pair_name = f\"{Path(cloth_img).stem}__{Path(person_img).stem}.png\"\n"
        if needle not in s:
            raise RuntimeError("Could not find expected pair_name assignment in CatVTON inference.py")
        if "if os.path.exists(os.path.join(output_dir, pair_name)):" not in s:
            s = s.replace(
                needle,
                needle + "            if os.path.exists(os.path.join(output_dir, pair_name)):\n                continue\n",
                1,
            )
        if marker_resume not in s:
            s = s.replace("# codex_multi_pairs_filename_patch_v1", f"# codex_multi_pairs_filename_patch_v1\n# {marker_resume}", 1)
        return s

    changed = _patch_once(path, marker=marker_resume, transform=transform_resume)
    if changed:
        print(f"[patch] CatVTON: enabled resume/skip-existing in {path}")


def _patch_hrviton(repo_dir: Path) -> None:
    """
    HR-VITON uses deprecated NumPy aliases (np.float/np.int) which are removed in NumPy>=1.24.
    Patch inference code to work with modern NumPy.
    """

    marker = "codex_numpy_alias_patch_v1"
    marker_progress = "codex_progress_bar_patch_v1"
    marker_grid = "codex_save_grid_optional_patch_v1"
    marker_skip = "codex_skip_existing_outputs_patch_v1"

    # test_generator.py (used by our runner)
    tg_path = repo_dir / "test_generator.py"

    def transform_tg(src: str) -> str:
        s = src
        s = s.replace("astype(np.float)", "astype(np.float32)")
        if marker not in s:
            s = s.replace("from utils import *", f"from utils import *\n\n# {marker}", 1)
        return s

    changed = _patch_once(tg_path, marker=marker, transform=transform_tg)
    if changed:
        print(f"[patch] HR-VITON: patched {tg_path}")

    def transform_tg_progress(src: str) -> str:
        s = src
        if marker_progress not in s:
            if "# codex_numpy_alias_patch_v1" in s:
                s = s.replace(
                    "# codex_numpy_alias_patch_v1",
                    "# codex_numpy_alias_patch_v1\n\n"
                    f"# {marker_progress}\n"
                    "try:\n"
                    "    from tqdm import tqdm  # type: ignore\n"
                    "except Exception:\n"
                    "    tqdm = None\n",
                    1,
                )
            else:
                # Fallback: insert near the top of file.
                s = s.replace(
                    "import time\n",
                    "import time\n\n"
                    f"# {marker_progress}\n"
                    "try:\n"
                    "    from tqdm import tqdm  # type: ignore\n"
                    "except Exception:\n"
                    "    tqdm = None\n\n",
                    1,
                )

        # Add a flag to optionally save visualization grids (very slow / huge for large runs).
        needle_outdir = 'parser.add_argument("--output_dir", type=str, default="./Output")'
        if needle_outdir in s and "--save_grid" not in s:
            s = s.replace(
                needle_outdir,
                needle_outdir
                + "\n    parser.add_argument('--save_grid', action='store_true', help='save visualization grids (very large)')",
                1,
            )

        # Make grid_dir respect --output_dir and only create/save when enabled.
        old_block = (
            "    if opt.output_dir is not None:\n"
            "        output_dir = opt.output_dir\n"
            "    else:\n"
            "        output_dir = os.path.join('./output', opt.test_name,\n"
            "                            opt.datamode, opt.datasetting, 'generator', 'output')\n"
            "    grid_dir = os.path.join('./output', opt.test_name,\n"
            "                             opt.datamode, opt.datasetting, 'generator', 'grid')\n"
            "    \n"
            "    os.makedirs(grid_dir, exist_ok=True)\n"
            "    \n"
            "    os.makedirs(output_dir, exist_ok=True)\n"
        )
        if old_block in s:
            new_block = (
                "    if opt.output_dir is not None:\n"
                "        output_dir = opt.output_dir\n"
                "    else:\n"
                "        output_dir = os.path.join('./output', opt.test_name,\n"
                "                            opt.datamode, opt.datasetting, 'generator', 'output')\n"
                "\n"
                "    os.makedirs(output_dir, exist_ok=True)\n"
                "    grid_dir = os.path.join(output_dir, 'grid')\n"
                "    if getattr(opt, 'save_grid', False):\n"
                "        os.makedirs(grid_dir, exist_ok=True)\n"
            )
            s = s.replace(old_block, new_block, 1)

        # Replace noisy per-step print with a progress bar (ETA) when available.
        if "print(num)" in s:
            s = s.replace("            print(num)\n", "", 1)

        # Insert progress bar setup right after iter_start_time.
        pb_setup = (
            "    iter_start_time = time.time()\n"
            "    with torch.no_grad():\n"
        )
        if pb_setup in s and "pbar =" not in s:
            s = s.replace(
                pb_setup,
                "    iter_start_time = time.time()\n"
                "    total = None\n"
                "    try:\n"
                "        total = len(test_loader.dataset)\n"
                "    except Exception:\n"
                "        pass\n"
                "    pbar = tqdm(total=total, unit='img') if tqdm is not None else None\n"
                "    with torch.no_grad():\n",
                1,
            )

        # Update progress bar after saving each batch output.
        update_needle = "            num += shape[0]\n"
        if update_needle in s and "pbar.update" not in s:
            s = s.replace(
                update_needle,
                update_needle + "            if pbar is not None:\n                pbar.update(shape[0])\n",
                1,
            )

        # Only build/save visualization grids if enabled.
        grid_save_needle = "                save_image(grid, os.path.join(grid_dir, unpaired_name))\n"
        if grid_save_needle in s and "if getattr(opt, 'save_grid'" not in s:
            s = s.replace(
                grid_save_needle,
                "                if getattr(opt, 'save_grid', False):\n"
                "                    grid = make_image_grid([(clothes[i].cpu() / 2 + 0.5), (pre_clothes_mask[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),\n"
                "                                            (warped_cloth[i].cpu().detach() / 2 + 0.5), (warped_clothmask[i].cpu().detach()).expand(3, -1, -1), visualize_segmap(fake_parse_gauss.cpu(), batch=i),\n"
                "                                            (pose_map[i].cpu()/2 +0.5), (warped_cloth[i].cpu()/2 + 0.5), (agnostic[i].cpu()/2 + 0.5),\n"
                "                                            (im[i]/2 +0.5), (output[i].cpu()/2 +0.5)],\n"
                "                                            nrow=4)\n"
                "                    save_image(grid, os.path.join(grid_dir, unpaired_name))\n",
                1,
            )
            # Remove the unconditional grid construction (the original 'grid = make_image_grid(...' line) if present.
            grid_construct = "                grid = make_image_grid(["
            if grid_construct in s:
                s = s.replace(
                    "                grid = make_image_grid([(clothes[i].cpu() / 2 + 0.5), (pre_clothes_mask[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),\n"
                    "                                        (warped_cloth[i].cpu().detach() / 2 + 0.5), (warped_clothmask[i].cpu().detach()).expand(3, -1, -1), visualize_segmap(fake_parse_gauss.cpu(), batch=i),\n"
                    "                                        (pose_map[i].cpu()/2 +0.5), (warped_cloth[i].cpu()/2 + 0.5), (agnostic[i].cpu()/2 + 0.5),\n"
                    "                                        (im[i]/2 +0.5), (output[i].cpu()/2 +0.5)],\n"
                    "                                        nrow=4)\n",
                    "",
                    1,
                )

        # Close progress bar at end of test().
        end_print = "    print(f\"Test time {time.time() - iter_start_time}\")\n"
        if end_print in s and "pbar.close" not in s:
            s = s.replace(
                end_print,
                "    if pbar is not None:\n"
                "        pbar.close()\n"
                + end_print,
                1,
            )
        return s

    changed = _patch_once(tg_path, marker=marker_progress, transform=transform_tg_progress)
    if changed:
        print(f"[patch] HR-VITON: added progress/ETA to {tg_path}")

    def transform_tg_grid_optional(src: str) -> str:
        lines = src.splitlines(keepends=True)

        # Find the visualization loop inside test(): "unpaired_names = []" then "for i in range(shape[0]):"
        start = None
        for i in range(len(lines) - 1):
            if lines[i].strip() == "unpaired_names = []" and "for i in range(shape[0]):" in lines[i + 1]:
                start = i
                break
        if start is None:
            raise RuntimeError("Could not find expected HR-VITON visualization loop to patch")

        # Find end of loop (the append line).
        end = None
        for j in range(start + 2, min(start + 80, len(lines))):
            if "unpaired_names.append(unpaired_name)" in lines[j]:
                end = j
                break
        if end is None:
            raise RuntimeError("Could not find end of HR-VITON visualization loop to patch")

        block = lines[start : end + 1]
        if any("if getattr(opt, 'save_grid'" in ln for ln in block):
            # Already patched; just add marker so we don't try again.
            out = src
        else:
            # Identify the lines inside the loop.
            # Expected order: grid = make_image_grid(...) [multi-line], unpaired_name = ..., save_image(...), append(...)
            idx_name = None
            idx_save = None
            for k, ln in enumerate(block):
                if "unpaired_name =" in ln:
                    idx_name = k
                if "save_image(" in ln:
                    idx_save = k
            if idx_name is None or idx_save is None:
                raise RuntimeError("Could not find unpaired_name/save_image lines in HR-VITON visualization loop")

            grid_block = block[2:idx_name]  # after "unpaired_names" + "for i ..." up to name line
            name_line = block[idx_name]
            save_line = block[idx_save]
            append_line = block[-1]

            indent = " " * 16  # existing indent inside the for-loop
            ind_if = indent + "if getattr(opt, 'save_grid', False):\n"
            grid_block_indented = []
            for ln in grid_block:
                grid_block_indented.append(" " * 20 + ln.lstrip())
            save_line_indented = " " * 20 + save_line.lstrip()

            new_block = [
                block[0],  # unpaired_names = []
                block[1],  # for i in range...
                name_line,
                ind_if,
                *grid_block_indented,
                save_line_indented,
                append_line,
            ]
            lines[start : end + 1] = new_block
            out = "".join(lines)

        # Add marker comment near the existing progress marker.
        marker_line = f"# {marker_grid}\n"
        if marker_line not in out:
            if f"# {marker_progress}" in out:
                out = out.replace(f"# {marker_progress}\n", f"# {marker_progress}\n{marker_line}", 1)
            else:
                out = marker_line + out
        return out

    changed = _patch_once(tg_path, marker=marker_grid, transform=transform_tg_grid_optional)
    if changed:
        print(f"[patch] HR-VITON: made --save_grid optional in {tg_path}")

    def transform_tg_skip_existing(src: str) -> str:
        s = src
        needle = "            shape = pre_clothes_mask.shape\n            \n            # multi-task inputs\n"
        if needle not in s:
            raise RuntimeError("Could not find expected insertion point in HR-VITON test_generator.py")
        insert = (
            "            shape = pre_clothes_mask.shape\n\n"
            f"            # {marker_skip}\n"
            "            try:\n"
            "                _out_names = []\n"
            "                for _i in range(shape[0]):\n"
            "                    _name = (inputs['c_name']['paired'][_i].split('.')[0] + '_' + inputs['c_name'][opt.datasetting][_i].split('.')[0] + '.png')\n"
            "                    _out_names.append(_name)\n"
            "                if _out_names and all(os.path.exists(os.path.join(output_dir, n)) for n in _out_names):\n"
            "                    num += shape[0]\n"
            "                    if pbar is not None:\n"
            "                        pbar.update(shape[0])\n"
            "                    continue\n"
            "            except Exception:\n"
            "                pass\n\n"
            "            # multi-task inputs\n"
        )
        return s.replace(needle, insert, 1)

    changed = _patch_once(tg_path, marker=marker_skip, transform=transform_tg_skip_existing)
    if changed:
        print(f"[patch] HR-VITON: enabled resume/skip-existing in {tg_path}")

    # utils.py (imported by test_generator.py)
    utils_path = repo_dir / "utils.py"

    def transform_utils(src: str) -> str:
        s = src
        s = s.replace("astype(np.int)", "astype(np.int32)")
        if marker not in s:
            s = s.replace("import numpy as np", f"import numpy as np\n# {marker}", 1)
        return s

    changed = _patch_once(utils_path, marker=marker, transform=transform_utils)
    if changed:
        print(f"[patch] HR-VITON: patched {utils_path}")


def _patch_stableviton(repo_dir: Path) -> None:
    """
    StableVITON prints coarse batch counters; add tqdm progress + ETA.
    """
    path = repo_dir / "inference.py"
    marker = "codex_progress_bar_patch_v1"

    def transform(src: str) -> str:
        s = src
        if marker not in s:
            needle = "from utils import tensor2img\n"
            if needle in s:
                s = s.replace(
                    needle,
                    needle
                    + "\n"
                    f"# {marker}\n"
                    "try:\n"
                    "    from tqdm import tqdm  # type: ignore\n"
                    "except Exception:\n"
                    "    tqdm = None\n",
                    1,
                )
            else:
                s = s.replace(
                    "import argparse\n",
                    "import argparse\n\n"
                    f"# {marker}\n"
                    "try:\n"
                    "    from tqdm import tqdm  # type: ignore\n"
                    "except Exception:\n"
                    "    tqdm = None\n\n",
                    1,
                )

        # Replace the main inference loop to use tqdm over images.
        old = (
            "    save_dir = opj(args.save_dir, \"unpair\" if args.unpair else \"pair\")\n"
            "    os.makedirs(save_dir, exist_ok=True)\n"
            "    for batch_idx, batch in enumerate(dataloader):\n"
            "        print(f\"{batch_idx}/{len(dataloader)}\")\n"
        )
        if old not in s:
            raise RuntimeError("Could not find expected StableVITON loop header to patch")
        new = (
            "    save_dir = opj(args.save_dir, \"unpair\" if args.unpair else \"pair\")\n"
            "    os.makedirs(save_dir, exist_ok=True)\n"
            "    pbar = tqdm(total=len(dataset), unit='img') if tqdm is not None else None\n"
            "    for batch_idx, batch in enumerate(dataloader):\n"
            "        if pbar is None:\n"
            "            print(f\"{batch_idx}/{len(dataloader)}\")\n"
        )
        s = s.replace(old, new, 1)

        # Update progress bar once per batch.
        update_needle = "        x_samples = model.decode_first_stage(samples)\n"
        if update_needle in s and "pbar.update(bs)" not in s:
            s = s.replace(update_needle, update_needle + "        if pbar is not None:\n            pbar.update(bs)\n", 1)

        end_needle = "            cv2.imwrite(to_path, x_sample_img[:,:,::-1])\n"
        if end_needle in s and "pbar.close()" not in s:
            s = s.replace(
                end_needle,
                end_needle + "\n    if pbar is not None:\n        pbar.close()\n",
                1,
            )
        return s

    changed = _patch_once(path, marker=marker, transform=transform)
    if changed:
        print(f"[patch] StableVITON: added progress/ETA to {path}")

    marker_skip = "codex_skip_existing_outputs_patch_v1"

    def transform_skip(src: str) -> str:
        s = src
        needle = "        z, c = model.get_input(batch, params.first_stage_key)\n"
        if needle not in s:
            raise RuntimeError("Could not find expected StableVITON get_input line to patch")
        insert = (
            "        # codex_skip_existing_outputs_patch_v1\n"
            "        try:\n"
            "            _out_paths = [\n"
            "                opj(save_dir, f\"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg\")\n"
            "                for fn, cloth_fn in zip(batch.get('img_fn', []), batch.get('cloth_fn', []))\n"
            "            ]\n"
            "            if _out_paths and all(os.path.exists(p) for p in _out_paths):\n"
            "                try:\n"
            "                    if pbar is not None:\n"
            "                        pbar.update(len(_out_paths))\n"
            "                except Exception:\n"
            "                    pass\n"
            "                continue\n"
            "        except Exception:\n"
            "            pass\n\n"
            "        z, c = model.get_input(batch, params.first_stage_key)\n"
        )
        return s.replace(needle, insert, 1)

    changed = _patch_once(path, marker=marker_skip, transform=transform_skip)
    if changed:
        print(f"[patch] StableVITON: enabled resume/skip-existing in {path}")

    attn_path = repo_dir / "ldm" / "modules" / "attention.py"
    marker_hint = "codex_crossattention_hint_patch_v1"

    def transform_attn(src: str) -> str:
        s = src
        if "class CrossAttention(nn.Module):" not in s:
            raise RuntimeError("Could not find CrossAttention class in StableVITON attention.py")
        s = s.replace("class CrossAttention(nn.Module):", f"class CrossAttention(nn.Module):\n    # {marker_hint}", 1)
        old_sig = "def forward(self, x, context=None, mask=None, mask1=None, mask2=None, use_attention_tv_loss=False):"
        if old_sig in s:
            s = s.replace(
                old_sig,
                "def forward(self, x, context=None, mask=None, hint=None, mask1=None, mask2=None, use_attention_tv_loss=False, **kwargs):",
                1,
            )
        return s

    changed = _patch_once(attn_path, marker=marker_hint, transform=transform_attn)
    if changed:
        print(f"[patch] StableVITON: patched CrossAttention.forward() signature in {attn_path}")


def _conda_run(
    env: str,
    args: Sequence[str],
    *,
    cwd: Path,
    dry_run: bool,
    ijit_stub: Optional[Path] = None,
) -> None:
    cmd = ["conda", "run", "-n", env]
    if ijit_stub is not None:
        existing = os.environ.get("LD_PRELOAD", "")
        preload = str(ijit_stub) + (f":{existing}" if existing else "")
        cmd += ["env", f"LD_PRELOAD={preload}"]
    cmd += list(args)
    if dry_run:
        print("+", " ".join(cmd))
        return
    _run(cmd, cwd=cwd)


def _ensure_ijit_stub(repo_root: Path, *, dry_run: bool) -> Optional[Path]:
    """
    Some conda PyTorch builds reference Intel JIT profiling symbols (iJIT_*) at load time.
    On machines without the corresponding runtime, importing torch fails with:
        libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent

    To keep this repo self-contained, we build a tiny stub shared library and LD_PRELOAD it for all baseline runs.
    """

    out_dir = repo_root / "external_vtons" / "runtime" / "ijit_stub"
    so_path = out_dir / "libijit_stub.so"
    src_path = out_dir / "ijit_stub.c"

    if so_path.exists():
        return so_path

    src = (
        "#include <stdint.h>\n\n"
        "// Minimal stubs for Intel JIT profiling symbols expected by some PyTorch builds.\n"
        "// Returning 0 disables profiling.\n\n"
        "int iJIT_NotifyEvent(int event_type, void *event_data) {\n"
        "    (void)event_type;\n"
        "    (void)event_data;\n"
        "    return 0;\n"
        "}\n\n"
        "unsigned int iJIT_GetNewMethodID(void) {\n"
        "    static unsigned int next_id = 1;\n"
        "    return next_id++;\n"
        "}\n\n"
        "int iJIT_IsProfilingActive(void) {\n"
        "    return 0;\n"
        "}\n"
    )

    if dry_run:
        print(f"[ijit_stub] would build: {so_path}", flush=True)
        return so_path

    out_dir.mkdir(parents=True, exist_ok=True)
    if not src_path.exists():
        src_path.write_text(src, encoding="utf-8")

    _run(["gcc", "-shared", "-fPIC", "-O2", "-o", str(so_path), str(src_path)])
    if not so_path.exists():
        return None
    return so_path


def _copy_tree_flat(src_dir: Path, dst_dir: Path, *, exts: Tuple[str, ...]) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        out = dst_dir / p.name
        if out.exists():
            continue
        shutil.copy2(p, out)
        n += 1
    return n


def _rename_hrviton_outputs(raw_out: Path, pred_dir: Path) -> int:
    """
    HR-VITON outputs files named: <person_id>_<cloth_id>.png (JPEG-encoded).
    We rename/copy them to: <cloth_id>__<person_id>.png to match our manifest ids.
    """
    pred_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in raw_out.glob("*.png"):
        # VITON-HD ids look like 00008_00; output stem is typically 00008_00_00006_00.
        parts = p.stem.split("_")
        if len(parts) < 4:
            continue
        person_id = "_".join(parts[:2])
        cloth_id = "_".join(parts[2:])
        out_name = f"{cloth_id}__{person_id}.png"
        out = pred_dir / out_name
        if out.exists():
            continue
        shutil.copy2(p, out)
        n += 1
    return n


def _rename_person_cloth_outputs(raw_out: Path, pred_dir: Path, *, exts: Tuple[str, ...]) -> int:
    """
    Rename/copy outputs named: <person_id>_<cloth_id>.<ext> (where ids themselves contain underscores)
    into our flat pred dir as: <cloth_id>__<person_id>.<ext>.
    """
    pred_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in raw_out.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        parts = p.stem.split("_")
        if len(parts) < 4:
            continue
        person_id = "_".join(parts[:2])
        cloth_id = "_".join(parts[2:])
        out_name = f"{cloth_id}__{person_id}{p.suffix.lower()}"
        out = pred_dir / out_name
        if out.exists():
            continue
        shutil.copy2(p, out)
        n += 1
    return n


def _collect_existing_outputs(baseline_key: str, *, raw_out: Path, pred_dir: Path) -> int:
    if baseline_key == "hrviton":
        return _rename_hrviton_outputs(raw_out, pred_dir)
    if baseline_key == "ladi":
        return _copy_tree_flat(raw_out, pred_dir, exts=IMAGE_EXTS)
    if baseline_key == "catvton":
        out_dir = raw_out / "vitonhd-512" / "unpaired"
        if not out_dir.exists():
            return 0
        return _copy_tree_flat(out_dir, pred_dir, exts=IMAGE_EXTS)
    if baseline_key == "stableviton":
        out_dir = raw_out / "unpair"
        if not out_dir.exists():
            return 0
        return _rename_person_cloth_outputs(out_dir, pred_dir, exts=(".jpg", ".jpeg", ".png"))
    return 0


def _score_folder(manifest: Path, pred_dir: Path, *, run_name: str, outdir: Path, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "evaluation.score_folder",
        "--manifest",
        str(manifest),
        "--pred-dir",
        str(pred_dir),
        "--outdir",
        str(outdir),
        "--run-name",
        run_name,
        "--no-metrics",
        "--save-triplets",
    ]
    if dry_run:
        print("+", " ".join(cmd))
        return
    _run(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run 5 external VTON baselines on the same (cloth_id, person_id) pair list extracted from mytest3curt.\n\n"
            "This script assumes you first prepared a benchmark bundle with:\n"
            "  python3 scripts/prepare_external_vton_benchmark.py --relative-paths\n\n"
            "Notes:\n"
            "  - Many external repos assume each person appears once; we patch a few of them to emit unique filenames.\n"
            "  - Some baselines require downloading checkpoints manually (HR-VITON, StableVITON, OOTDiffusion).\n"
        )
    )
    p.add_argument("--bench-root", default="external_vtons/bench_mytest3curt_ok", help="Benchmark root created by prepare_external_vton_benchmark.py")
    p.add_argument("--pred-root", default="external_vtons/preds_mytest3curt_ok", help="Where to write flat pred folders per baseline")
    p.add_argument("--runs-outdir", default="runs", help="Where to write scored run folders (evaluation.score_folder)")
    p.add_argument(
        "--baselines",
        default="hrviton,ladi,idm,catvton,stableviton",
        help="Comma-separated baselines to run: hrviton,ladi,idm,catvton,stableviton,ootdiffusion",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    p.add_argument(
        "--overwrite-preds",
        action="store_true",
        help="Delete existing pred/raw output folders for each baseline before running (dangerous).",
    )
    p.add_argument("--score", action="store_true", help="After running each baseline, score it into runs/ via evaluation.score_folder")
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip baseline inference if pred folder already looks complete (default: true).",
    )
    p.add_argument(
        "--skip-score-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip scoring if an identical scored run already exists in --runs-outdir (default: true).",
    )
    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue running other baselines if one fails (default: true).",
    )
    p.add_argument(
        "--skip-if-low-vram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip very heavy baselines when free VRAM is too low (default: true).",
    )
    p.add_argument(
        "--parent-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print parent-side progress/ETA by counting output files during inference (default: true).",
    )
    p.add_argument(
        "--parent-progress-interval-s",
        type=float,
        default=30.0,
        help="Seconds between parent-side progress updates (default: 30).",
    )
    p.add_argument(
        "--min-free-vram-idm-mib",
        type=int,
        default=12000,
        help="Minimum free VRAM (MiB) required to attempt IDM-VTON (default: 12000).",
    )
    p.add_argument(
        "--min-free-vram-stableviton-mib",
        type=int,
        default=12000,
        help="Minimum free VRAM (MiB) required to attempt StableVITON (default: 12000).",
    )
    p.add_argument(
        "--run-name-prefix",
        default="baseline",
        help="Prefix used for scored run names: <prefix>_<baseline> (default: baseline).",
    )

    # Conda env names (create them yourself, or edit to match your setup).
    p.add_argument("--env-hrviton", default="hrviton", help="Conda env for HR-VITON")
    p.add_argument("--env-ladi", default="ladi-vton", help="Conda env for LaDI-VTON")
    p.add_argument("--env-idm", default="idm", help="Conda env for IDM-VTON")
    p.add_argument("--env-catvton", default="catvton", help="Conda env for CatVTON")
    p.add_argument("--env-stableviton", default="stableviton", help="Conda env for StableVITON")
    p.add_argument("--env-ootdiffusion", default="ootd", help="Conda env for OOTDiffusion")

    # HR-VITON checkpoints (required if you run hrviton).
    p.add_argument("--hrviton-tocg", default="", help="Path to HR-VITON TOCG checkpoint (.pth)")
    p.add_argument("--hrviton-gen", default="", help="Path to HR-VITON generator checkpoint (.pth)")

    # LaDI model id (some defaults are gated on Hugging Face).
    p.add_argument(
        "--ladi-pretrained-model",
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Hugging Face model id/path for LaDI-VTON --pretrained_model_name_or_path.",
    )

    # IDM-VTON settings (memory heavy).
    p.add_argument("--idm-test-batch-size", type=int, default=1, help="IDM-VTON test batch size (default: 1)")
    p.add_argument(
        "--idm-mixed-precision",
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="IDM-VTON mixed precision (default: fp16)",
    )
    p.add_argument(
        "--idm-xformers",
        action="store_true",
        help="Enable xFormers in IDM-VTON (requires xformers installed in that env).",
    )

    p.add_argument("--catvton-batch-size", type=int, default=4, help="CatVTON batch size (default: 4)")

    # StableVITON checkpoint (required if you run stableviton).
    p.add_argument("--stableviton-ckpt", default="", help="Path to StableVITON checkpoint (.ckpt/.pth)")
    p.add_argument(
        "--stableviton-config",
        default="configs/VITONHD.yaml",
        help="StableVITON config path (relative to repo) (default: configs/VITONHD.yaml).",
    )
    p.add_argument("--stableviton-img-h", type=int, default=1024, help="StableVITON inference height (default: 1024)")
    p.add_argument("--stableviton-img-w", type=int, default=768, help="StableVITON inference width (default: 768)")
    p.add_argument("--stableviton-batch-size", type=int, default=1, help="StableVITON batch size (default: 1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    ijit_stub = _ensure_ijit_stub(repo_root, dry_run=args.dry_run)

    bench_root = (repo_root / args.bench_root).resolve()
    manifest = bench_root / "manifest.jsonl"
    dataroot = bench_root / "dataroot"
    if not manifest.exists() or not dataroot.exists():
        raise SystemExit(
            f"Benchmark missing. Run:\n  python3 scripts/prepare_external_vton_benchmark.py --relative-paths\n"
            f"Expected: {manifest} and {dataroot}"
        )

    pred_root = (repo_root / args.pred_root).resolve()
    runs_outdir = (repo_root / args.runs_outdir).resolve()
    pred_root.mkdir(parents=True, exist_ok=True)

    expected_samples = _count_nonempty_lines(manifest)
    manifest_sha256 = _sha256_file(manifest)
    failures: List[str] = []

    repos_root = repo_root / "external_vtons" / "repos"
    ckpts_root = repo_root / "external_vtons" / "checkpoints"

    def _pick_ckpt(arg_value: str, candidates: Sequence[Path]) -> Optional[Path]:
        if arg_value:
            p = Path(arg_value).expanduser()
            return p
        for c in candidates:
            if c.exists():
                return c
        return None

    hrviton_tocg_ckpt = _pick_ckpt(
        str(args.hrviton_tocg),
        [
            ckpts_root / "hrviton" / "mtviton.pth",
            ckpts_root / "hrviton" / "tocg.pth",
        ],
    )
    hrviton_gen_ckpt = _pick_ckpt(
        str(args.hrviton_gen),
        [
            ckpts_root / "hrviton" / "gen.pth",
            ckpts_root / "hrviton" / "generator.pth",
        ],
    )
    stableviton_ckpt = _pick_ckpt(
        str(args.stableviton_ckpt),
        [
            ckpts_root / "stableviton" / "VITONHD.ckpt",
        ],
    )

    baselines: Dict[str, Baseline] = {
        "hrviton": Baseline(
            key="hrviton",
            repo_url="https://github.com/sangyun884/HR-VITON.git",
            repo_dir=repos_root / "hrviton",
            conda_env=args.env_hrviton,
        ),
        "ladi": Baseline(
            key="ladi",
            repo_url="https://github.com/miccunifi/ladi-vton.git",
            repo_dir=repos_root / "ladi-vton",
            conda_env=args.env_ladi,
        ),
        "idm": Baseline(
            key="idm",
            repo_url="https://github.com/yisol/IDM-VTON.git",
            repo_dir=repos_root / "idm-vton",
            conda_env=args.env_idm,
        ),
        "catvton": Baseline(
            key="catvton",
            repo_url="https://github.com/Zheng-Chong/CatVTON.git",
            repo_dir=repos_root / "catvton",
            conda_env=args.env_catvton,
        ),
        "stableviton": Baseline(
            key="stableviton",
            repo_url="https://github.com/rlawjdghek/StableVITON.git",
            repo_dir=repos_root / "stableviton",
            conda_env=args.env_stableviton,
        ),
        "ootdiffusion": Baseline(
            key="ootdiffusion",
            repo_url="https://github.com/levihsu/OOTDiffusion.git",
            repo_dir=repos_root / "ootdiffusion",
            conda_env=args.env_ootdiffusion,
        ),
    }

    want = [b.strip() for b in str(args.baselines).split(",") if b.strip()]
    unknown = [b for b in want if b not in baselines]
    if unknown:
        raise SystemExit(f"Unknown baselines: {unknown}. Known: {sorted(baselines.keys())}")

    for key in want:
        b = baselines[key]
        _ensure_repo(b.repo_dir, b.repo_url, dry_run=args.dry_run)

        # Apply minimal patches for multi-pair evaluation where needed.
        if not args.dry_run:
            if b.key == "hrviton":
                _patch_hrviton(b.repo_dir)
            elif b.key == "ladi":
                _patch_ladi(b.repo_dir)
            elif b.key == "idm":
                _patch_idm(b.repo_dir)
            elif b.key == "catvton":
                _patch_catvton(b.repo_dir)
            elif b.key == "stableviton":
                _patch_stableviton(b.repo_dir)

        pred_dir = pred_root / b.key / "pred"
        raw_out = pred_root / b.key / "raw"
        if args.overwrite_preds and (not args.dry_run):
            if pred_dir.exists():
                shutil.rmtree(pred_dir)
            if raw_out.exists():
                shutil.rmtree(raw_out)
        pred_dir.mkdir(parents=True, exist_ok=True)
        raw_out.mkdir(parents=True, exist_ok=True)

        # If this is a rerun, first try to collect any already-generated outputs into pred_dir.
        if (not args.dry_run) and _count_images(pred_dir, recursive=False) < expected_samples:
            n = _collect_existing_outputs(b.key, raw_out=raw_out, pred_dir=pred_dir)
            if n:
                print(f"[{b.key}] collected {n} existing preds into {pred_dir}", flush=True)

        pred_count = _count_images(pred_dir, recursive=False)
        if args.skip_existing and (pred_count >= expected_samples) and (not args.dry_run):
            print(f"[skip][{b.key}] preds complete ({pred_count}/{expected_samples}); skipping inference", flush=True)
        else:

            if b.key == "hrviton":
                if not hrviton_tocg_ckpt or not hrviton_gen_ckpt:
                    print(
                        "[skip][hrviton] Missing checkpoints. Provide --hrviton-tocg/--hrviton-gen "
                        "or download to external_vtons/checkpoints/hrviton/.",
                        flush=True,
                    )
                    continue
                try:
                    mon: Optional[Tuple[threading.Event, threading.Thread]] = None
                    mon = _start_parent_progress_monitor(
                        enabled=bool(args.parent_progress) and (not args.dry_run),
                        baseline_key=b.key,
                        watch_dir=raw_out,
                        expected=expected_samples,
                        recursive=False,
                        interval_s=float(args.parent_progress_interval_s),
                    )
                    _conda_run(
                        b.conda_env,
                        [
                            "python",
                            "test_generator.py",
                            "--occlusion",
                            "--cuda",
                            "True",
                            "--gpu_ids",
                            "0",
                            "--datasetting",
                            "unpaired",
                            "--dataroot",
                            str(dataroot),
                            "--datamode",
                            "test",
                            "--data_list",
                            "test_pairs.txt",
                            "--output_dir",
                            str(raw_out),
                            "--tocg_checkpoint",
                            str(hrviton_tocg_ckpt),
                            "--gen_checkpoint",
                            str(hrviton_gen_ckpt),
                            "--fine_height",
                            "1024",
                            "--fine_width",
                            "768",
                            "--batch-size",
                            "1",
                        ],
                        cwd=b.repo_dir,
                        dry_run=args.dry_run,
                        ijit_stub=ijit_stub,
                    )
                except subprocess.CalledProcessError:
                    if not args.dry_run:
                        _collect_existing_outputs(b.key, raw_out=raw_out, pred_dir=pred_dir)
                    failures.append(b.key)
                    print(f"[error][{b.key}] inference failed", flush=True)
                    if not args.continue_on_error:
                        raise
                    continue
                finally:
                    _stop_parent_progress_monitor(mon)
                if not args.dry_run:
                    n = _collect_existing_outputs(b.key, raw_out=raw_out, pred_dir=pred_dir)
                    print(f"[hrviton] collected {n} preds into {pred_dir}")

            elif b.key == "ladi":
                if (
                    (not args.dry_run)
                    and str(args.ladi_pretrained_model).strip() == "stabilityai/stable-diffusion-2-inpainting"
                    and (not _has_hf_token())
                ):
                    print(
                        "[skip][ladi] Hugging Face token not found for gated model "
                        "'stabilityai/stable-diffusion-2-inpainting'.\n"
                        "  Fix: conda run -n ladi-vton huggingface-cli login (and accept the model license),\n"
                        "  or pass --ladi-pretrained-model <your_model_id_or_local_path>.",
                        flush=True,
                    )
                    continue
                try:
                    mon: Optional[Tuple[threading.Event, threading.Thread]] = None
                    mon = _start_parent_progress_monitor(
                        enabled=bool(args.parent_progress) and (not args.dry_run),
                        baseline_key=b.key,
                        watch_dir=raw_out,
                        expected=expected_samples,
                        recursive=True,
                        interval_s=float(args.parent_progress_interval_s),
                    )
                    _conda_run(
                        b.conda_env,
                        [
                            "python",
                            "src/inference.py",
                            "--pretrained_model_name_or_path",
                            str(args.ladi_pretrained_model),
                            "--dataset",
                            "vitonhd",
                            "--vitonhd_dataroot",
                            str(dataroot),
                            "--output_dir",
                            str(raw_out),
                            "--test_order",
                            "unpaired",
                            "--category",
                            "all",
                            "--mixed_precision",
                            "fp16",
                            "--use_png",
                        ],
                        cwd=b.repo_dir,
                        dry_run=args.dry_run,
                        ijit_stub=ijit_stub,
                    )
                except subprocess.CalledProcessError:
                    if not args.dry_run:
                        _collect_existing_outputs(b.key, raw_out=raw_out, pred_dir=pred_dir)
                    failures.append(b.key)
                    print(
                        "[error][ladi] inference failed (often HF auth/gated model). "
                        "Try: conda run -n ladi-vton huggingface-cli login, or pass "
                        "--ladi-pretrained-model runwayml/stable-diffusion-inpainting",
                        flush=True,
                    )
                    if not args.continue_on_error:
                        raise
                    continue
                finally:
                    _stop_parent_progress_monitor(mon)
                if not args.dry_run:
                    n = _collect_existing_outputs(b.key, raw_out=raw_out, pred_dir=pred_dir)
                    print(f"[ladi] collected {n} preds into {pred_dir}")

            elif b.key == "idm":
                if args.skip_if_low_vram and (not args.dry_run):
                    mem = _try_query_gpu_mem()
                    if mem is not None and mem.free_mib < int(args.min_free_vram_idm_mib):
                        print(
                            f"[skip][idm] low free VRAM ({mem.free_mib}/{mem.total_mib} MiB free). "
                            "Close other GPU processes (e.g. ComfyUI) and rerun, or use --no-skip-if-low-vram.",
                            flush=True,
                        )
                        for line in _try_query_gpu_processes():
                            print(f"[gpu] {line}", flush=True)
                        continue

                # Ensure tagged JSON files exist in dataroot/test (IDM-VTON expects them).
                if not args.dry_run:
                    (dataroot / "test").mkdir(parents=True, exist_ok=True)
                    src_test = b.repo_dir / "vitonhd_test_tagged.json"
                    src_train = b.repo_dir / "vitonhd_train_tagged.json"
                    if src_test.exists():
                        shutil.copy2(src_test, dataroot / "test" / "vitonhd_test_tagged.json")
                    if src_train.exists():
                        (dataroot / "train").mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_train, dataroot / "train" / "vitonhd_train_tagged.json")

                try:
                    mon: Optional[Tuple[threading.Event, threading.Thread]] = None
                    mon = _start_parent_progress_monitor(
                        enabled=bool(args.parent_progress) and (not args.dry_run),
                        baseline_key=b.key,
                        watch_dir=pred_dir,
                        expected=expected_samples,
                        recursive=False,
                        interval_s=float(args.parent_progress_interval_s),
                    )
                    idm_args = [
                        "accelerate",
                        "launch",
                        "inference.py",
                        "--width",
                        "768",
                        "--height",
                        "1024",
                        "--num_inference_steps",
                        "30",
                        "--output_dir",
                        str(pred_dir),
                        "--unpaired",
                        "--data_dir",
                        str(dataroot),
                        "--seed",
                        "42",
                        "--test_batch_size",
                        str(args.idm_test_batch_size),
                        "--guidance_scale",
                        "2.0",
                        "--mixed_precision",
                        str(args.idm_mixed_precision),
                    ]
                    if args.idm_xformers:
                        idm_args.append("--enable_xformers_memory_efficient_attention")
                    _conda_run(
                        b.conda_env,
                        idm_args,
                        cwd=b.repo_dir,
                        dry_run=args.dry_run,
                        ijit_stub=ijit_stub,
                    )
                except subprocess.CalledProcessError:
                    failures.append(b.key)
                    print(f"[error][{b.key}] inference failed", flush=True)
                    if not args.continue_on_error:
                        raise
                    continue
                finally:
                    _stop_parent_progress_monitor(mon)

            elif b.key == "catvton":
                try:
                    mon: Optional[Tuple[threading.Event, threading.Thread]] = None
                    mon = _start_parent_progress_monitor(
                        enabled=bool(args.parent_progress) and (not args.dry_run),
                        baseline_key=b.key,
                        watch_dir=raw_out,
                        expected=expected_samples,
                        recursive=True,
                        interval_s=float(args.parent_progress_interval_s),
                    )
                    _conda_run(
                        b.conda_env,
                        [
                            "python",
                            "inference.py",
                            "--dataset_name",
                            "vitonhd",
                            "--data_root_path",
                            str(dataroot),
                            "--output_dir",
                            str(raw_out),
                            "--height",
                            "512",
                            "--width",
                            "384",
                            "--batch_size",
                            str(args.catvton_batch_size),
                            "--mixed_precision",
                            "fp16",
                            "--num_inference_steps",
                            "30",
                            "--guidance_scale",
                            "2.5",
                        ],
                        cwd=b.repo_dir,
                        dry_run=args.dry_run,
                        ijit_stub=ijit_stub,
                    )
                except subprocess.CalledProcessError:
                    if not args.dry_run:
                        _collect_existing_outputs(b.key, raw_out=raw_out, pred_dir=pred_dir)
                    failures.append(b.key)
                    print(f"[error][{b.key}] inference failed", flush=True)
                    if not args.continue_on_error:
                        raise
                    continue
                finally:
                    _stop_parent_progress_monitor(mon)
                if not args.dry_run:
                    n = _collect_existing_outputs(b.key, raw_out=raw_out, pred_dir=pred_dir)
                    print(f"[catvton] collected {n} preds into {pred_dir}")

            elif b.key == "stableviton":
                if not stableviton_ckpt:
                    print(
                        "[skip][stableviton] Missing checkpoint. Provide --stableviton-ckpt "
                        "or download to external_vtons/checkpoints/stableviton/VITONHD.ckpt.",
                        flush=True,
                    )
                    continue
                if args.skip_if_low_vram and (not args.dry_run):
                    mem = _try_query_gpu_mem()
                    if mem is not None and mem.free_mib < int(args.min_free_vram_stableviton_mib):
                        print(
                            f"[skip][stableviton] low free VRAM ({mem.free_mib}/{mem.total_mib} MiB free). "
                            "Close other GPU processes and rerun, or use --no-skip-if-low-vram.",
                            flush=True,
                        )
                        for line in _try_query_gpu_processes():
                            print(f"[gpu] {line}", flush=True)
                        continue
                try:
                    mon: Optional[Tuple[threading.Event, threading.Thread]] = None
                    mon = _start_parent_progress_monitor(
                        enabled=bool(args.parent_progress) and (not args.dry_run),
                        baseline_key=b.key,
                        watch_dir=raw_out,
                        expected=expected_samples,
                        recursive=True,
                        interval_s=float(args.parent_progress_interval_s),
                    )
                    _conda_run(
                        b.conda_env,
                        [
                            "python",
                            "inference.py",
                            "--config_path",
                            str(Path(args.stableviton_config)),
                            "--batch_size",
                            str(args.stableviton_batch_size),
                            "--model_load_path",
                            str(stableviton_ckpt),
                            "--data_root_dir",
                            str(dataroot),
                            "--unpair",
                            "--save_dir",
                            str(raw_out),
                            "--img_H",
                            str(args.stableviton_img_h),
                            "--img_W",
                            str(args.stableviton_img_w),
                        ],
                        cwd=b.repo_dir,
                        dry_run=args.dry_run,
                        ijit_stub=ijit_stub,
                    )
                except subprocess.CalledProcessError:
                    if not args.dry_run:
                        _collect_existing_outputs(b.key, raw_out=raw_out, pred_dir=pred_dir)
                    failures.append(b.key)
                    print(
                        f"[error][{b.key}] inference failed (often CUDA OOM). "
                        "Try closing other GPU processes (e.g. ComfyUI) or reduce batch size.",
                        flush=True,
                    )
                    if not args.continue_on_error:
                        raise
                    continue
                finally:
                    _stop_parent_progress_monitor(mon)
                if not args.dry_run:
                    n = _collect_existing_outputs(b.key, raw_out=raw_out, pred_dir=pred_dir)
                    print(f"[stableviton] collected {n} preds into {pred_dir}")

            elif b.key == "ootdiffusion":
                print(
                    "[skip][ootdiffusion] Not auto-run yet. "
                    "OOTDiffusion needs HF checkpoints under <repo>/checkpoints/ (see its README).",
                    flush=True,
                )

            else:
                raise AssertionError(f"Unhandled baseline key: {b.key}")

        if args.score and (not args.dry_run):
            pred_count = _count_images(pred_dir, recursive=False)
            if pred_count < expected_samples:
                print(
                    f"[skip][score][{b.key}] preds incomplete ({pred_count}/{expected_samples}); skipping scoring",
                    flush=True,
                )
                continue
            run_name = f"{args.run_name_prefix}_{b.key}"
            if args.skip_score_existing:
                existing = _find_existing_scored_run(
                    runs_outdir,
                    run_name=run_name,
                    manifest_sha256=manifest_sha256,
                    pred_dir=pred_dir,
                    expected_samples=expected_samples,
                )
                if existing is not None:
                    print(f"[skip][score][{b.key}] already scored: {existing.name}", flush=True)
                else:
                    _score_folder(
                        manifest,
                        pred_dir,
                        run_name=run_name,
                        outdir=runs_outdir,
                        dry_run=args.dry_run,
                    )
            else:
                _score_folder(
                    manifest,
                    pred_dir,
                    run_name=run_name,
                    outdir=runs_outdir,
                    dry_run=args.dry_run,
                )

    if failures:
        print(f"[done] baselines failed: {', '.join(failures)}", flush=True)


if __name__ == "__main__":
    main()
