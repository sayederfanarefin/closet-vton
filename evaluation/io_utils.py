import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image_rgb(path: Path) -> torch.Tensor:
    """Returns float tensor in [0,1] with shape (3,H,W)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def load_image_rgb_from_bytes(image_bytes: bytes) -> torch.Tensor:
    """Returns float tensor in [0,1] with shape (3,H,W)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def resize_rgb_tensor(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Bilinear resize for float RGB tensor (3,H,W) in [0,1]."""
    if t.shape[1] == target_h and t.shape[2] == target_w:
        return t
    x = t[None, ...]
    x = torch.nn.functional.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return x[0]


def load_mask(path: Path, threshold: float = 0.5) -> torch.Tensor:
    """Returns float tensor mask in {0,1} with shape (1,H,W)."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    m = (arr >= threshold).astype(np.float32)
    t = torch.from_numpy(m)[None, ...].contiguous()
    return t


def save_image_rgb(t: torch.Tensor, path: Path) -> None:
    """Expects float tensor in [0,1], shape (3,H,W)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = t.detach().clamp(0, 1)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_comparison_grid(
    *,
    person: torch.Tensor,
    cloth: torch.Tensor,
    pred: torch.Tensor,
    gt: Optional[torch.Tensor],
    path: Path,
    label_person: str = "person",
    label_cloth: str = "cloth",
    label_pred: str = "pred",
    label_gt: str = "gt",
) -> None:
    """Creates a simple 2x2 grid PNG for quick visual inspection."""
    tiles: Tuple[Tuple[Optional[torch.Tensor], str], ...] = (
        (person, label_person),
        (cloth, label_cloth),
        (pred, label_pred),
        (gt, label_gt),
    )

    def to_pil(x: Optional[torch.Tensor]) -> Image.Image:
        if x is None:
            return Image.new("RGB", (256, 256), color=(240, 240, 240))
        x = x.detach().clamp(0, 1)
        arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    pil_tiles = [(to_pil(t), name) for t, name in tiles]
    # Use the max width/height among tiles so we don't distort aspect ratios.
    max_w = max(im.width for im, _ in pil_tiles)
    max_h = max(im.height for im, _ in pil_tiles)

    def pad(im: Image.Image) -> Image.Image:
        canvas = Image.new("RGB", (max_w, max_h), color=(0, 0, 0))
        x0 = (max_w - im.width) // 2
        y0 = (max_h - im.height) // 2
        canvas.paste(im, (x0, y0))
        return canvas

    a, b, c, d = [pad(im) for im, _ in pil_tiles]
    grid = Image.new("RGB", (max_w * 2, max_h * 2), color=(0, 0, 0))
    grid.paste(a, (0, 0))
    grid.paste(b, (max_w, 0))
    grid.paste(c, (0, max_h))
    grid.paste(d, (max_w, max_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def extract_first_base64_image(response: Dict[str, Any]) -> Tuple[bytes, str]:
    """
    Returns (image_bytes, mime_type).

    Handles a few possible ComfyUI websocket payload shapes; adjust if your
    ComfyUI build returns a different schema.
    """
    images = None
    if isinstance(response.get("images"), list):
        images = response["images"]
    elif isinstance(response.get("data"), dict) and isinstance(response["data"].get("images"), list):
        images = response["data"]["images"]
    elif (
        isinstance(response.get("data"), dict)
        and isinstance(response["data"].get("result"), dict)
        and isinstance(response["data"]["result"].get("images"), list)
    ):
        images = response["data"]["result"]["images"]

    if not images:
        raise ValueError(f"Could not find images array in response keys={list(response.keys())}")

    first = images[0]
    mime = "image/png"
    if isinstance(first, dict):
        b64 = first.get("base64") or first.get("image_base64") or first.get("b64")
        mime = first.get("type") or first.get("mime") or mime
        if b64 is None and isinstance(first.get("data"), str):
            b64 = first["data"]
    elif isinstance(first, str):
        b64 = first
    else:
        raise ValueError(f"Unsupported image entry type: {type(first)}")

    if not isinstance(b64, str) or b64.strip() == "":
        raise ValueError("Image base64 payload missing/empty")

    b64 = b64.strip()
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]

    return base64.b64decode(b64), mime


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True)
