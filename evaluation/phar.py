from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class PharConfig:
    # Final aggregation weights (must sum to 1.0)
    w_human: float = 0.35
    w_id: float = 0.25
    w_gar: float = 0.25
    w_fit: float = 0.15

    # CLIP-based critic calibration
    clip_model: str = "openai/clip-vit-base-patch32"
    critic_tau: float = 0.20
    critic_prompts_real: str = "a real photograph of a person"
    critic_prompts_fake: str = "an ai-generated image of a person"

    # Garment fidelity weights
    w_color: float = 0.5
    w_texture: float = 0.5
    color_sigma: float = 20.0  # exp(-dE / sigma)

    # Fit consistency
    pose_lambda: float = 4.0  # exp(-lambda * mean_kpt_dist)
    halo_alpha: float = 1.5  # exp(-alpha * halo_excess)

    # Cropping parameters (relative to image size)
    face_expand: float = 1.8
    hand_expand: float = 2.2


class _ClipHelper:
    def __init__(self, model_name: str, device: torch.device):
        from transformers import CLIPModel, CLIPProcessor  # local import to keep base deps minimal

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Cache text embeddings for common prompts
        self._text_cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _to_pil(img: torch.Tensor):
        from PIL import Image

        x = img.detach().clamp(0, 1)
        arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    @torch.no_grad()
    def embed_image(self, img: torch.Tensor) -> torch.Tensor:
        pil = self._to_pil(img)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self.model.get_image_features(**inputs)
        feat = F.normalize(feat, dim=-1)
        return feat[0]

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        cached = self._text_cache.get(text)
        if cached is not None:
            return cached
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self.model.get_text_features(**inputs)
        feat = F.normalize(feat, dim=-1)
        out = feat[0]
        self._text_cache[text] = out
        return out

    def cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a * b).sum().clamp(-1, 1).item())

    def critic_score(self, img: torch.Tensor, real_prompt: str, fake_prompt: str, tau: float) -> float:
        e_img = self.embed_image(img)
        e_real = self.embed_text(real_prompt)
        e_fake = self.embed_text(fake_prompt)
        s_real = self.cosine(e_img, e_real)
        s_fake = self.cosine(e_img, e_fake)
        # Map to [0,1] via a calibrated sigmoid on the similarity gap.
        x = torch.tensor((s_real - s_fake) / max(tau, 1e-6))
        return float(torch.sigmoid(x).item())


class _PoseHelper:
    """
    Keypoint-based helper used for:
    - face crop (identity + human realism critic)
    - hand crops (human realism critic)
    - pose drift (fit consistency)
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            import torchvision
            from torchvision.models.detection import keypointrcnn_resnet50_fpn

            # Torchvision API differs across versions; handle both.
            weights = None
            try:
                from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

                weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
                self.model = keypointrcnn_resnet50_fpn(weights=weights)
            except Exception:
                self.model = keypointrcnn_resnet50_fpn(pretrained=True)  # type: ignore[arg-type]

            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to initialize torchvision KeypointRCNN. "
                "Install/upgrade torchvision or disable PHaR pose features."
            ) from exc

    @torch.no_grad()
    def infer(self, img: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        # model expects list[Tensor(C,H,W)] in [0,1]
        x = img.to(self.device)
        out = self.model([x])[0]
        scores = out.get("scores")
        if scores is None or scores.numel() == 0:
            return None
        best = int(torch.argmax(scores).item())
        if float(scores[best].item()) < 0.2:
            return None
        return {
            "box": out["boxes"][best],  # (4,)
            "keypoints": out["keypoints"][best],  # (17,3) (x,y,score)
            "score": scores[best],
        }

    @staticmethod
    def _clip_box(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = box
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        return x0, y0, x1, y1

    def face_crop(self, img: torch.Tensor, det: Dict[str, torch.Tensor], expand: float) -> Optional[torch.Tensor]:
        k = det["keypoints"]  # (17,3)
        # COCO keypoints: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear
        pts = k[[0, 1, 2, 3, 4], :2]
        conf = k[[0, 1, 2, 3, 4], 2]
        valid = conf > 0.2
        if valid.sum().item() < 2:
            return None
        pts = pts[valid]
        x0, y0 = pts.min(dim=0).values
        x1, y1 = pts.max(dim=0).values

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        w_box = (x1 - x0).clamp_min(10.0) * expand
        h_box = (y1 - y0).clamp_min(10.0) * expand * 1.25

        x0i = int((cx - w_box / 2).round().item())
        x1i = int((cx + w_box / 2).round().item())
        y0i = int((cy - h_box / 2).round().item())
        y1i = int((cy + h_box / 2).round().item())
        _, h, w = img.shape
        x0i, y0i, x1i, y1i = self._clip_box((x0i, y0i, x1i, y1i), w=w, h=h)
        return img[:, y0i:y1i, x0i:x1i]

    def hand_crops(self, img: torch.Tensor, det: Dict[str, torch.Tensor], expand: float) -> List[torch.Tensor]:
        k = det["keypoints"]
        # wrists: 9 left_wrist, 10 right_wrist
        wrists = k[[9, 10]]
        conf = wrists[:, 2]
        pts = wrists[:, :2]
        # scale crop size based on person box
        box = det["box"]
        box_w = (box[2] - box[0]).clamp_min(50.0)
        box_h = (box[3] - box[1]).clamp_min(50.0)
        side = float(torch.sqrt(box_w * box_h).item()) * 0.25 * expand

        crops: List[torch.Tensor] = []
        _, h, w = img.shape
        for i in range(2):
            if float(conf[i].item()) < 0.2:
                continue
            cx, cy = pts[i]
            x0 = int((cx - side / 2).round().item())
            x1 = int((cx + side / 2).round().item())
            y0 = int((cy - side / 2).round().item())
            y1 = int((cy + side / 2).round().item())
            x0, y0, x1, y1 = self._clip_box((x0, y0, x1, y1), w=w, h=h)
            crops.append(img[:, y0:y1, x0:x1])
        return crops

    def pose_distance(self, det_a: Dict[str, torch.Tensor], det_b: Dict[str, torch.Tensor], w: int, h: int) -> Optional[float]:
        ka = det_a["keypoints"]  # (17,3)
        kb = det_b["keypoints"]
        conf = (ka[:, 2] > 0.2) & (kb[:, 2] > 0.2)
        if conf.sum().item() < 5:
            return None
        da = ka[conf][:, :2]
        db = kb[conf][:, :2]
        # Normalize by image diagonal so scores are comparable across resolutions.
        diag = float((w * w + h * h) ** 0.5)
        dist = torch.norm((da - db) / max(diag, 1.0), dim=1)
        return float(dist.mean().item())


def _masked_mean_lab(img: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    x = img.detach().clamp(0, 1)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    if mask is None:
        return lab.reshape(-1, 3).mean(axis=0)
    m = mask.detach().cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    m = (m >= 0.5).astype(np.uint8)
    if m.sum() < 10:
        return lab.reshape(-1, 3).mean(axis=0)
    vals = lab[m.astype(bool)]
    return vals.mean(axis=0)


def _mask_bbox(mask: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    m = mask.detach().cpu()
    if m.ndim == 3:
        m = m[0]
    ys, xs = torch.where(m >= 0.5)
    if ys.numel() == 0:
        return None
    y0 = int(ys.min().item())
    y1 = int(ys.max().item()) + 1
    x0 = int(xs.min().item())
    x1 = int(xs.max().item()) + 1
    return x0, y0, x1, y1


def _masked_crop(img: torch.Tensor, mask: Optional[torch.Tensor], pad_frac: float = 0.10) -> torch.Tensor:
    if mask is None:
        return img
    bbox = _mask_bbox(mask)
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    _, h, w = img.shape
    pad_x = int((x1 - x0) * pad_frac)
    pad_y = int((y1 - y0) * pad_frac)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    return img[:, y0:y1, x0:x1]


def _boundary_band(mask: torch.Tensor, radius: int = 5) -> torch.Tensor:
    """
    Returns a (1,H,W) float mask in {0,1} representing a thin band around the boundary.
    """
    m = (mask >= 0.5).float()
    if m.ndim == 2:
        m = m[None, ...]
    # Dilate via max-pool
    k = 2 * radius + 1
    dil = F.max_pool2d(m[None, ...], kernel_size=k, stride=1, padding=radius)[0]
    # Erode via dilation of inverted mask
    ero = 1.0 - F.max_pool2d((1.0 - m)[None, ...], kernel_size=k, stride=1, padding=radius)[0]
    band = (dil - ero).clamp(0, 1)
    return band


def _halo_score(pred: torch.Tensor, garment_mask: torch.Tensor, alpha: float) -> float:
    """
    Heuristic halo detector:
    - compute Laplacian magnitude on grayscale
    - measure average magnitude in a boundary band vs global
    - penalize excessive boundary energy (\"cutout\" halos)
    """
    band = _boundary_band(garment_mask, radius=5).to(pred.device)
    gray = (0.2989 * pred[0] + 0.5870 * pred[1] + 0.1140 * pred[2])[None, None, ...]
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=pred.dtype, device=pred.device)[None, None, ...]
    lap = F.conv2d(gray, kernel, padding=1).abs()[0, 0]
    band_m = band[0]
    band_mean = (lap * band_m).sum() / band_m.sum().clamp_min(1.0)
    global_mean = lap.mean().clamp_min(1e-6)
    ratio = (band_mean / global_mean).clamp_min(1.0)
    # excess above 1.0 gets penalized
    excess = (ratio - 1.0).item()
    score = float(torch.exp(torch.tensor(-alpha * excess)).clamp(0, 1).item())
    return score


class PharScorer:
    """
    PHaR: Physically-grounded Human Apparel Realism score for VTON outputs.

    This implementation is fully automated and conditional on (person, garment, output).
    Some sub-components use practical proxies (e.g., CLIP-based critic) so the
    metric is runnable end-to-end without bespoke training.
    """

    def __init__(self, cfg: PharConfig, device: Optional[str] = None, enable_pose: bool = True):
        self.cfg = cfg
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.clip = _ClipHelper(cfg.clip_model, self.device)
        self.pose = _PoseHelper(self.device) if enable_pose else None

    def score(
        self,
        *,
        person: torch.Tensor,
        garment: torch.Tensor,
        pred: torch.Tensor,
        garment_mask: Optional[torch.Tensor] = None,
        garment_image_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        cfg = self.cfg

        def _resize_mask_to(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
            m = mask
            if m.ndim == 2:
                m = m[None, ...]
            if m.shape[1] == h and m.shape[2] == w:
                return m
            m = m[None, ...].float()
            m = F.interpolate(m, size=(h, w), mode="nearest")
            return m[0]

        # Move images to device for torch ops (CLIP helper moves internally as needed)
        person_d = person.to(self.device)
        garment_d = garment.to(self.device)
        pred_d = pred.to(self.device)

        # Ensure optional masks align with their corresponding images.
        if garment_mask is not None:
            garment_mask = _resize_mask_to(garment_mask, pred.shape[1], pred.shape[2])
        if garment_image_mask is not None:
            garment_image_mask = _resize_mask_to(garment_image_mask, garment.shape[1], garment.shape[2])

        # --------------------------
        # A) Photographic human realism
        # --------------------------
        # Use a CLIP-based critic: real-photo vs ai-generated prompt gap.
        crops_for_human: List[Tuple[torch.Tensor, float]] = []
        crops_for_human.append((pred, 0.5))

        det_pred = None
        if self.pose is not None:
            det_pred = self.pose.infer(pred)
            if det_pred is not None:
                face = self.pose.face_crop(pred, det_pred, expand=cfg.face_expand)
                if face is not None:
                    crops_for_human.append((face, 0.3))
                hands = self.pose.hand_crops(pred, det_pred, expand=cfg.hand_expand)
                if hands:
                    w_hand = 0.2 / float(len(hands))
                    for hand in hands:
                        crops_for_human.append((hand, w_hand))

        # Renormalize weights if some crops are missing.
        wsum = sum(w for _, w in crops_for_human)
        if wsum <= 0:
            r_human = 0.5
        else:
            r_human = 0.0
            for img, w in crops_for_human:
                r_human += (w / wsum) * self.clip.critic_score(
                    img,
                    cfg.critic_prompts_real,
                    cfg.critic_prompts_fake,
                    cfg.critic_tau,
                )

        # --------------------------
        # B) Identity preservation
        # --------------------------
        # Primary: face crop similarity (CLIP image embedding proxy).
        id_score = 0.5
        if self.pose is not None:
            det_person = self.pose.infer(person)
            if det_person is not None and det_pred is not None:
                face_p = self.pose.face_crop(person, det_person, expand=cfg.face_expand)
                face_o = self.pose.face_crop(pred, det_pred, expand=cfg.face_expand)
                if face_p is not None and face_o is not None:
                    e_p = self.clip.embed_image(face_p)
                    e_o = self.clip.embed_image(face_o)
                    sim = self.clip.cosine(e_p, e_o)
                    id_score = float((sim + 1.0) / 2.0)
                else:
                    # fallback: full-image similarity (weaker proxy)
                    e_p = self.clip.embed_image(person)
                    e_o = self.clip.embed_image(pred)
                    sim = self.clip.cosine(e_p, e_o)
                    id_score = float((sim + 1.0) / 2.0) * 0.7
        else:
            e_p = self.clip.embed_image(person)
            e_o = self.clip.embed_image(pred)
            sim = self.clip.cosine(e_p, e_o)
            id_score = float((sim + 1.0) / 2.0) * 0.7

        id_score = float(np.clip(id_score, 0.0, 1.0))

        # --------------------------
        # C) Garment fidelity
        # --------------------------
        # Color term (LAB mean delta) + texture term (CLIP similarity on masked crops).
        color_score = 0.5
        mu_g = _masked_mean_lab(garment, garment_image_mask)
        mu_o = _masked_mean_lab(pred, garment_mask)
        if mu_g is not None and mu_o is not None:
            dE = float(np.linalg.norm(mu_g - mu_o))
            color_score = float(np.exp(-dE / max(cfg.color_sigma, 1e-6)))

        g_crop = _masked_crop(garment, garment_image_mask)
        o_crop = _masked_crop(pred, garment_mask)
        e_g = self.clip.embed_image(g_crop)
        e_o = self.clip.embed_image(o_crop)
        tex_sim = self.clip.cosine(e_g, e_o)
        texture_score = float((tex_sim + 1.0) / 2.0)

        f_gar = cfg.w_color * float(np.clip(color_score, 0.0, 1.0)) + cfg.w_texture * float(np.clip(texture_score, 0.0, 1.0))
        f_gar = float(np.clip(f_gar, 0.0, 1.0))

        # --------------------------
        # D) Physical fit & occlusion consistency
        # --------------------------
        # Proxy: pose stability + boundary halo penalty.
        pose_score = 0.5
        if self.pose is not None:
            det_person = self.pose.infer(person)
            if det_person is not None and det_pred is not None:
                _, h, w = pred.shape
                dist = self.pose.pose_distance(det_person, det_pred, w=w, h=h)
                if dist is not None:
                    pose_score = float(np.exp(-cfg.pose_lambda * dist))

        halo = 1.0
        if garment_mask is not None:
            try:
                halo = _halo_score(pred_d, garment_mask.to(self.device), alpha=cfg.halo_alpha)
            except Exception:
                halo = 1.0

        c_fit = float(np.clip(pose_score * halo, 0.0, 1.0))

        # --------------------------
        # Final PHaR
        # --------------------------
        phar = 100.0 * (
            cfg.w_human * r_human +
            cfg.w_id * id_score +
            cfg.w_gar * f_gar +
            cfg.w_fit * c_fit
        )
        phar = float(np.clip(phar, 0.0, 100.0))

        return {
            "phar": phar,
            "phar_human": float(r_human),
            "phar_id": float(id_score),
            "phar_gar": float(f_gar),
            "phar_fit": float(c_fit),
            "phar_gar_color": float(color_score),
            "phar_gar_texture": float(texture_score),
            "phar_fit_pose": float(pose_score),
            "phar_fit_halo": float(halo),
        }
