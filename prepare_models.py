import json
import os
import requests
from tqdm import tqdm
from datetime import datetime

try:
    # Used only for lightweight validation of existing VAE files.
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safe_open = None

# Adjust URLs based on actual model sources
MODEL_SOURCES = {
    "exp-schp-201908301523-atr.pth": "https://huggingface.co/zhengchong/CatVTON/resolve/2969fcf85fe62f2036605716f0b56f0b81d01d79/SCHP/exp-schp-201908301523-atr.pth",
    # NOTE: Flux models require the Flux autoencoder (AE), not a Stable Diffusion VAE.
    # Public mirror of FLUX.1-dev AE (saved as ae.sft in many ComfyUI workflows):
    #   - https://huggingface.co/pretentioushorsefly/flux-models
    "ae.sft": "https://huggingface.co/pretentioushorsefly/flux-models/resolve/main/models/vae/ae.safetensors",
    "flux1-redux-dev.safetensors": "https://huggingface.co/second-state/FLUX.1-Redux-dev-GGUF/resolve/c7e36ea59a409eaa553b9744b53aa350099d5d51/flux1-redux-dev.safetensors",
    "sigclip_vision_patch14_384.safetensors": "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors",
    "flux1-fill-dev-fp16-Q8_0-GGUF.gguf": "https://huggingface.co/SporkySporkness/FLUX.1-Fill-dev-GGUF/resolve/main/flux1-fill-dev-fp16-Q8_0-GGUF.gguf",
    "t5-v1_1-xxl-encoder-Q6_K.gguf": "https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q6_K.gguf",
#clip_l.safetensors: https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors
    "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
    "sam_vit_b (375MB)": "https://huggingface.co/segments-arnaud/sam_vit_b/resolve/f38484d6934e5d2b555b1685d22d676236455685/sam_vit_b_01ec64.pth",
    "GroundingDINO_SwinT_OGC (694MB)": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    "diffusion_pytorch_model.safetensors": "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union/resolve/967982231cc032f743d7867d8eeb7dd44391c35a/diffusion_pytorch_model.safetensors",
    "pytorch_lora_weights.safetensors": "https://huggingface.co/linoyts/huggy_v29/resolve/main/pytorch_lora_weights.safetensors"
}

def _dest_subdir(node_class_type: str, input_name: str) -> str:
    # Keep this mapping conservative: if we guess wrong, ComfyUI won't see the file.
    # These are standard ComfyUI model folders (under COMFYUI_MODELS_DIR).
    if input_name == "vae_name":
        return "vae"
    if input_name == "style_model_name":
        return "style_models"
    if input_name == "unet_name":
        return "unet"
    if input_name in ("clip_name1", "clip_name2"):
        return "clip"
    if input_name == "clip_name":
        # Disambiguate CLIP-Vision models vs text encoders.
        if node_class_type == "CLIPVisionLoader":
            return "clip_vision"
        return "clip"
    if input_name == "lora_name":
        return "loras"

    # Segment Anything extension weights. The extension may manage its own downloads;
    # we place them in a predictable location anyway.
    if node_class_type.startswith("SAMModelLoader"):
        return "sam"
    if node_class_type.startswith("GroundingDinoModelLoader"):
        return "groundingdino"

    return ""

def download(url, filepath):
    headers = {}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(filepath, 'wb') as f, tqdm(
        desc=os.path.basename(filepath),
        total=total,
        unit='B',
        unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))

def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _is_likely_sd_vae(path: str) -> bool:
    """
    Heuristic: Stable Diffusion VAEs commonly use 4 latent channels.
    Flux.1 uses 16 latent channels, so if we detect a 4-channel SD-style VAE
    at the expected Flux AE path, we should replace it.
    """
    if safe_open is None:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            # SD VAEs usually have this key and it reveals latent channels.
            if "decoder.conv_in.weight" in f.keys():
                w = f.get_tensor("decoder.conv_in.weight")
                # shape: [out_channels, latent_channels, kH, kW]
                if len(w.shape) == 4 and int(w.shape[1]) == 4:
                    return True
    except Exception:
        # If we can't read it as safetensors, it's not what we want here.
        return True
    return False

def prepare_models(workflow_file):
    with open(workflow_file, 'r') as file:
        workflow = json.load(file)

    # (model_name, dest_subdir)
    models_needed = set()

    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get('inputs', {})
        node_class_type = node.get("class_type", "")
        if node_class_type == "AdaptiveClothingSegment":
            models_needed.add(("exp-schp-201908301523-atr.pth", "human_parsing"))
        for k, value in inputs.items():
            if not (isinstance(value, str) and value in MODEL_SOURCES):
                continue
            dest = _dest_subdir(str(node_class_type), str(k))
            models_needed.add((value, dest))

    comfyui_dir = os.getenv("COMFYUI_DIR", "/opt/ComfyUI")
    model_dir = os.getenv("COMFYUI_MODELS_DIR", os.path.join(comfyui_dir, "models"))
    os.makedirs(model_dir, exist_ok=True)

    for model, dest in sorted(models_needed):
        url = MODEL_SOURCES[model]
        target_dir = os.path.join(model_dir, dest) if dest else model_dir
        os.makedirs(target_dir, exist_ok=True)
        outpath = os.path.join(target_dir, model)
        # Migrate from older "flat" layout (models/<file>) to subfolders to avoid re-downloads.
        legacy_path = os.path.join(model_dir, model)
        if dest and not os.path.exists(outpath) and os.path.exists(legacy_path) and os.path.getsize(legacy_path) > 0:
            print(f"[prepare_models] moving: {legacy_path} -> {outpath}")
            os.replace(legacy_path, outpath)
            continue
        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            # Special-case: ae.sft historically gets mistakenly populated with an SD VAE.
            # If we detect that, back it up and download the proper Flux AE.
            if model == "ae.sft" and dest == "vae" and _is_likely_sd_vae(outpath):
                bak = f"{outpath}.bak_{_timestamp()}_sdvae"
                print(f"[prepare_models] WARNING: {outpath} looks like an SD VAE (latent_channels=4).")
                print(f"[prepare_models] backing up: {outpath} -> {bak}")
                os.replace(outpath, bak)
            else:
                print(f"[prepare_models] exists, skipping: {outpath}")
                continue
        print(f"[prepare_models] downloading: {model} -> {outpath}")
        download(url, outpath)

if __name__ == "__main__":
    prepare_models('workflow_api.json')
