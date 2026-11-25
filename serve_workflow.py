# serve_workflow.py
import os
import json
import secrets
import uuid
import base64
import copy
import shutil
import time
import threading
import socket
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import RootModel
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode, urlparse, urlunparse
from websocket import create_connection, WebSocketTimeoutException, WebSocket
import requests
from PIL import Image, UnidentifiedImageError

app = FastAPI()

# Simple in-memory job store for async requests
JOB_STORE: Dict[str, Dict[str, Any]] = {}
JOB_LOCK = threading.Lock()
REQUEST_LOG_LOCK = threading.Lock()

# Cache ComfyUI /object_info so we can fill new required inputs (ComfyUI updates can add required fields).
_OBJECT_INFO: Optional[Dict[str, Any]] = None
_OBJECT_INFO_AT: float = 0.0
_OBJECT_INFO_LOCK = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load the workflow JSON at startup
# ─────────────────────────────────────────────────────────────────────────────

WORKFLOW_PATH = os.getenv(
    "WORKFLOW_PATH",
    os.path.join(os.path.dirname(__file__), "workflow_api.json"),
)
try:
    with open(WORKFLOW_PATH, "r") as f:
        WORKFLOW_JSON = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Could not find workflow_api.json at {WORKFLOW_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 2) HTTP Basic Auth setup
# ─────────────────────────────────────────────────────────────────────────────

security = HTTPBasic()
API_USER = os.getenv("API_USER", "admin")
API_PASS = os.getenv("API_PASS", "changeme")

def check_auth(creds: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(creds.username, API_USER)
    correct_password = secrets.compare_digest(creds.password, API_PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return creds.username

# ─────────────────────────────────────────────────────────────────────────────
# 3) Accept arbitrary JSON overrides via RootModel (if provided as a string)
# ─────────────────────────────────────────────────────────────────────────────

class JSONOverrides(RootModel[dict]):
    root: dict

# ─────────────────────────────────────────────────────────────────────────────
# 4) /generate endpoint (protected by Basic Auth; accepts two image files plus optional overrides)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _set_job(job_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        job = JOB_STORE.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = _now_iso()

def _parse_overrides(overrides: Optional[str]) -> Dict[str, Any]:
    if not overrides:
        return {}
    try:
        parsed = json.loads(overrides)
        if not isinstance(parsed, dict):
            raise ValueError("`overrides` must be a JSON object mapping node IDs to dicts")
        return parsed
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in 'overrides': {e}"
        )

def _save_uploads(file50: UploadFile, file59: UploadFile) -> Dict[str, str]:
    # ComfyUI's stock LoadImage node reads from its input directory (usually /opt/ComfyUI/input in our container).
    # Allow override via env var for custom setups.
    save_dir = os.getenv("COMFYUI_IMAGES_DIR", "/opt/ComfyUI/input")
    os.makedirs(save_dir, exist_ok=True)

    def save_upload_to_uuid(upload: UploadFile) -> str:
        _, ext = os.path.splitext(upload.filename or "")
        if ext == "":
            ext = ".png"
        unique_name = f"{uuid.uuid4().hex}{ext}"
        dest_path = os.path.join(save_dir, unique_name)
        # Be defensive: depending on client/lib, UploadFile's pointer may not be at 0.
        try:
            upload.file.seek(0)
        except Exception:
            pass
        with open(dest_path, "wb") as f_out:
            shutil.copyfileobj(upload.file, f_out)
            f_out.flush()
            os.fsync(f_out.fileno())

        # Fast failure with an actionable error (ComfyUI would later crash at LoadImage).
        if os.path.getsize(dest_path) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{upload.filename}' saved as empty file (0 bytes).",
            )
        try:
            with Image.open(dest_path) as im:
                im.verify()
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{upload.filename}' is not a valid image (PIL could not identify it).",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{upload.filename}' failed image validation: {e}",
            )
        return unique_name

    print(f"[serve_workflow][DBG] Saving uploads to {save_dir}...")
    try:
        filename50 = save_upload_to_uuid(file50)
        filename59 = save_upload_to_uuid(file59)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed saving uploaded images: {e}"
        )
    finally:
        file50.file.close()
        file59.file.close()

    return {"50": filename50, "59": filename59}

def _comfy_http_base() -> str:
    # Prefer explicit HTTP base; otherwise derive from WS URL.
    http_url = os.getenv("COMFYUI_HTTP_URL", "").strip()
    if http_url:
        return http_url.rstrip("/")

    ws_url = os.getenv("COMFYUI_WS_URL", "ws://localhost:8188/ws").strip()
    parsed = urlparse(ws_url)
    scheme = "http" if parsed.scheme == "ws" else "https" if parsed.scheme == "wss" else "http"
    return urlunparse((scheme, parsed.netloc, "", "", "", "")).rstrip("/")

def _comfy_ws_url(client_id: str) -> str:
    ws_url = os.getenv("COMFYUI_WS_URL", "ws://localhost:8188/ws").strip()
    sep = "&" if "?" in ws_url else "?"
    return f"{ws_url}{sep}{urlencode({'clientId': client_id})}"

def _apply_inputs_to_workflow(workflow: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    # ComfyUI prompt API expects the workflow as a single dict ("prompt") where each node
    # contains an "inputs" dict. We patch those inputs per request.
    wf = copy.deepcopy(workflow)
    for node_id, overrides in (inputs or {}).items():
        nid = str(node_id)
        node = wf.get(nid)
        if node is None:
            continue
        if not isinstance(overrides, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid overrides for node {nid}: expected object, got {type(overrides)}",
            )
        node_inputs = node.get("inputs")
        if not isinstance(node_inputs, dict):
            node_inputs = {}
            node["inputs"] = node_inputs
        node_inputs.update(overrides)
    return wf

def _get_object_info(http_base: str) -> Dict[str, Any]:
    global _OBJECT_INFO, _OBJECT_INFO_AT
    ttl = float(os.getenv("COMFYUI_OBJECT_INFO_TTL_SEC", "300"))
    now = time.time()
    with _OBJECT_INFO_LOCK:
        if _OBJECT_INFO is not None and (now - _OBJECT_INFO_AT) < ttl:
            return _OBJECT_INFO

    r = requests.get(f"{http_base}/object_info", timeout=10)
    r.raise_for_status()
    obj = r.json()
    if not isinstance(obj, dict):
        raise RuntimeError("Unexpected /object_info response type")

    with _OBJECT_INFO_LOCK:
        _OBJECT_INFO = obj
        _OBJECT_INFO_AT = now
    return obj

def _spec_default(spec: Any, input_name: Optional[str] = None) -> Any:
    # /object_info "required" spec is typically:
    #   ["INT", {"default": 20, ...}]
    #   ["FLOAT", {"default": 1.0, ...}]
    #   [[choice1, choice2, ...], {"default": choice1, ...}]
    #
    # Some nodes omit the config dict, so we also handle:
    #   ["INT"]
    #   [[choice1, choice2, ...]]
    #
    # For connection types (MODEL, IMAGE, ...), there is no meaningful default.
    if spec is None:
        return None
    if isinstance(spec, dict):
        # Not typical for ComfyUI, but be tolerant.
        return spec.get("default")
    if not isinstance(spec, list) or len(spec) == 0:
        return None

    kind = spec[0]
    cfg = spec[1] if len(spec) >= 2 and isinstance(spec[1], dict) else {}
    if isinstance(cfg, dict) and "default" in cfg:
        return cfg["default"]

    # Enum / dropdown.
    if isinstance(kind, list) and kind:
        return kind[0]

    # Last resort: heuristic defaults for primitive numeric/bool inputs.
    if isinstance(kind, str):
        t = kind.upper()
        if t in {"FLOAT", "NUMBER"}:
            if input_name and any(k in input_name for k in ("strength", "scale", "weight")):
                return 1.0
            return 0.0
        if t in {"INT"}:
            return 0
        if t in {"BOOLEAN", "BOOL"}:
            return False

    return None

def _fill_missing_required_inputs(prompt: Dict[str, Any], http_base: str) -> Dict[str, Any]:
    # Some custom nodes/ComfyUI versions add new required fields over time
    # (e.g. INPAINT_ExpandMask.blur_type, StyleModelApply.strength).
    # Fill any missing required inputs that have a default in /object_info.
    obj = _get_object_info(http_base)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type")
        if not isinstance(ct, str) or not ct:
            continue
        schema = obj.get(ct)
        if not isinstance(schema, dict):
            continue
        required = (schema.get("input") or {}).get("required")
        if not isinstance(required, dict):
            continue
        node_inputs = node.get("inputs")
        if not isinstance(node_inputs, dict):
            node_inputs = {}
            node["inputs"] = node_inputs

        for input_name, spec in required.items():
            if input_name in node_inputs:
                continue
            default = _spec_default(spec, input_name=str(input_name))
            if default is None:
                continue
            node_inputs[input_name] = default
    return prompt

def _execute_workflow(inputs: Dict[str, Any]) -> Dict[str, Any]:
    execution_started = time.perf_counter()
    stage_latency: Dict[str, float] = {}
    segmentation: Dict[str, Any] = {}
    active_node = None
    active_started = execution_started
    connect_timeout = float(os.getenv("COMFYUI_CONNECT_TIMEOUT_SEC", "10"))
    recv_timeout = float(os.getenv("COMFYUI_RECV_TIMEOUT_SEC", "600"))

    http_base = _comfy_http_base()
    client_id = uuid.uuid4().hex
    ws_url = _comfy_ws_url(client_id)

    prompt = _apply_inputs_to_workflow(WORKFLOW_JSON, inputs)
    prompt = _fill_missing_required_inputs(prompt, http_base)

    # 1) Queue the prompt via ComfyUI's HTTP API
    try:
        r = requests.post(
            f"{http_base}/prompt",
            json={"prompt": prompt, "client_id": client_id},
            timeout=connect_timeout,
        )
        if r.status_code >= 400:
            # ComfyUI returns useful JSON for invalid prompts (e.g., missing custom nodes).
            body = (r.text or "").strip()
            raise RuntimeError(f"HTTP {r.status_code} body={body}")
        j = r.json()
        prompt_id = j.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"Missing prompt_id in response: {j}")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not queue prompt to ComfyUI at {http_base}: {exc}",
        )

    # 2) Listen for completion on the websocket (so we don't poll history blindly)
    try:
        ws: WebSocket = create_connection(ws_url, timeout=connect_timeout)
        ws.settimeout(recv_timeout)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not connect to ComfyUI WebSocket at {ws_url}: {exc}",
        )
    print(f"[serve_workflow][DBG] Connected to ComfyUI WebSocket at {ws_url}")

    start_time = time.time()
    print(f"[serve_workflow][DBG] Waiting for ComfyUI completion... prompt_id={prompt_id}")
    try:
        while True:
            raw = ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            parsed = json.loads(raw)

            msg_type = parsed.get("type")
            if msg_type == "executed":
                data = parsed.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    for value in (data.get("output") or {}).get("segmentation_fallback", []):
                        diag = json.loads(value)
                        segmentation[diag["role"]] = diag
                continue
            if msg_type == "status":
                queue_rem = parsed.get("data", {}).get("status", {}).get("exec_info", {}).get("queue_remaining", None)
                print(f"[serve_workflow] Status update: queue_remaining={queue_rem}")
                continue

            if msg_type == "execution_cached":
                data = parsed.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    for nid in data.get("nodes", []):
                        stage_latency[f"node_{nid}"] = 0.0
                continue

            if msg_type in {"execution_error", "execution_interrupted"}:
                data = parsed.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    if active_node is not None:
                        key = f"node_{active_node}"
                        stage_latency[key] = stage_latency.get(key, 0.0) + time.perf_counter() - active_started
                    rejection = None
                    message = str(data.get("exception_message", ""))
                    if "CROP_VALIDITY_REJECTED " in message:
                        try:
                            rejection = json.JSONDecoder().raw_decode(message.split("CROP_VALIDITY_REJECTED ", 1)[1])[0]
                        except (ValueError, TypeError):
                            pass
                    if rejection and rejection.get("role"):
                        segmentation[rejection["role"]] = rejection
                    raise HTTPException(
                        status_code=422 if rejection else 500,
                        detail={"error": "ComfyUI execution failed", "comfy_error": data,
                                "validity": rejection, "stage_latency": stage_latency,
                                "segmentation": segmentation},
                    )
                continue

            if msg_type == "executing":
                data = parsed.get("data", {}) or {}
                if data.get("prompt_id") != prompt_id:
                    continue
                now = time.perf_counter()
                if active_node is not None:
                    key = f"node_{active_node}"
                    stage_latency[key] = stage_latency.get(key, 0.0) + now - active_started
                active_node = data.get("node")
                active_started = now
                # When node is None, ComfyUI signals completion for that prompt.
                if data.get("node") is None:
                    break

    except WebSocketTimeoutException:
        elapsed = time.time() - start_time
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Timeout waiting for ComfyUI result after {elapsed:.1f}s",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error receiving from ComfyUI: {exc}",
        )
    finally:
        try:
            ws.close()
        except Exception:
            pass

    # 3) Fetch result metadata from history, then download the image bytes via /view
    try:
        hr = requests.get(f"{http_base}/history/{prompt_id}", timeout=30)
        hr.raise_for_status()
        history = hr.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed fetching ComfyUI history for prompt_id={prompt_id}: {exc}",
        )

    entry = history.get(prompt_id) if isinstance(history, dict) and prompt_id in history else history
    outputs = entry.get("outputs") if isinstance(entry, dict) else None
    if not isinstance(outputs, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected ComfyUI history format for prompt_id={prompt_id}",
        )
    for output in outputs.values():
        for value in output.get("segmentation_fallback", []):
            diag = json.loads(value)
            segmentation[diag["role"]] = diag

    image_info: Optional[Dict[str, Any]] = None
    # Prefer final output nodes (PreviewImage / SaveImage) if present.
    preferred_node_ids: List[str] = []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type")
        if ct in {"PreviewImage", "SaveImage"}:
            preferred_node_ids.append(str(nid))

    def _pick_image_from_output(out: Any) -> Optional[Dict[str, Any]]:
        imgs = out.get("images") if isinstance(out, dict) else None
        if isinstance(imgs, list) and imgs and isinstance(imgs[0], dict):
            return imgs[0]
        return None

    for nid in preferred_node_ids:
        out = outputs.get(nid)
        image_info = _pick_image_from_output(out)
        if image_info is not None:
            break

    # Never substitute an intermediate/reference output for the final person crop.
    validity = None
    for nid, node in prompt.items():
        if node.get("class_type") == "ValidatePersonCrop":
            values = outputs.get(str(nid), {}).get("crop_validity", [])
            if values:
                validity = json.loads(values[0])
            if not validity or not validity.get("valid"):
                raise HTTPException(status_code=422, detail="Final person crop did not pass validity checking")

    if image_info is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No images found in ComfyUI history for prompt_id={prompt_id}",
        )

    filename = image_info.get("filename")
    subfolder = image_info.get("subfolder", "")
    img_type = image_info.get("type", "output")
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid image metadata in ComfyUI history for prompt_id={prompt_id}: {image_info}",
        )

    try:
        view_url = f"{http_base}/view?{urlencode({'filename': filename, 'subfolder': subfolder, 'type': img_type})}"
        ir = requests.get(view_url, timeout=60)
        ir.raise_for_status()
        content = ir.content
        mime = (ir.headers.get("Content-Type") or "image/png").split(";", 1)[0].strip() or "image/png"
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed downloading image from ComfyUI view for prompt_id={prompt_id}: {exc}",
        )

    b64 = base64.b64encode(content).decode("ascii")
    return {
        "prompt_id": prompt_id,
        "images": [{"base64": b64, "type": mime}],
        "validity": validity,
        "segmentation": segmentation,
        "stage_latency": stage_latency,
        "service_elapsed_sec": time.perf_counter() - execution_started,
    }

def _run_job(job_id: str, inputs: Dict[str, Any], attempt_number: int = 1) -> None:
    started = time.perf_counter()
    record = {"job_id": job_id, "timestamp": _now_iso(), "worker": socket.gethostname(),
              "gpu": os.getenv("CROPFIX_GPU_NAME", "not_recorded"), "attempt_number": attempt_number}
    _set_job(job_id, status="running")
    try:
        result = _execute_workflow(inputs)
        _set_job(job_id, status="succeeded", result=result)
        record.update(status="succeeded", validity=result.get("validity"),
                      stage_latency=result.get("stage_latency"), segmentation=result.get("segmentation"))
    except HTTPException as exc:
        _set_job(job_id, status="failed", error=f"{exc.status_code}: {exc.detail}")
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        record.update(status="failed", error=str(exc.detail), validity=detail.get("validity"),
                      stage_latency=detail.get("stage_latency"), segmentation=detail.get("segmentation"))
    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc))
        record.update(status="failed", error=str(exc), validity=None)
    finally:
        record["elapsed_sec"] = time.perf_counter() - started
        validity = record.get("validity") or {}
        record["rho"] = validity.get("rho")
        record["valid"] = bool(validity.get("valid", False))
        _set_job(job_id, telemetry=record)
        path = os.getenv("CROPFIX_REQUEST_LOG", "")
        if path:
            with REQUEST_LOG_LOCK:
                with open(path, "a") as out:
                    out.write(json.dumps(record) + "\n")

@app.post("/generate")
async def generate(
    # (A) Enforce HTTP Basic Auth
    username: str = Depends(check_auth),

    # (B) Two image uploads for node 50 and node 59
    file50: UploadFile = File(..., description="Image file for node 50 (ClothesImage)"),
    file59: UploadFile = File(..., description="Image file for node 59 (PeoplePhoto)"),

    # (C) Optional JSON‐string field “overrides” with other node‐ID overrides
    overrides: Optional[str] = Form(None, description="JSON string of other node‐overrides")
):
    """
    Expects a multipart/form-data POST with:
      - file50: image for node '50'
      - file59: image for node '59'
      - overrides: (optional) JSON string mapping other node IDs to their field overrides.
    
    Example curl:
      curl -X POST http://localhost:8000/generate \
        -u myusername:mypassword \
        -F "file50=@/path/to/first_input.jpeg" \
        -F "file59=@/path/to/second_input.jpeg" \
        -F 'overrides={"3":{"seed":12345,"steps":10}}'
    """
    print (f"[serve_workflow][DBG] Received request from user '{username}'")
    # ─────────────────────────────────────────────────────────────────────────
    # (1) Parse any JSON overrides from the “overrides” form field
    # ─────────────────────────────────────────────────────────────────────────
    user_overrides: Dict[str, Any] = _parse_overrides(overrides)
    print(f"[serve_workflow][DBG] User overrides: {json.dumps(user_overrides)}")
    # ─────────────────────────────────────────────────────────────────────────
    # (2) Save the two uploaded files under /opt/ComfyUI/images/ using UUID filenames
    # ─────────────────────────────────────────────────────────────────────────
    filenames = _save_uploads(file50, file59)

    # ─────────────────────────────────────────────────────────────────────────
    # (3) Merge the image overrides into the overall inputs dict
    # ─────────────────────────────────────────────────────────────────────────
    inputs = {
        "50": {"image": filenames["50"]},
        "59": {"image": filenames["59"]},
        **user_overrides
    }
    print(f"[serve_workflow][DBG] Inputs for ComfyUI: {json.dumps(inputs)}")        
    return _execute_workflow(inputs)

@app.post("/generate_async")
async def generate_async(
    username: str = Depends(check_auth),
    file50: UploadFile = File(..., description="Image file for node 50 (ClothesImage)"),
    file59: UploadFile = File(..., description="Image file for node 59 (PeoplePhoto)"),
    overrides: Optional[str] = Form(None, description="JSON string of other node-overrides"),
    attempt_number: int = Form(1, ge=1, le=10),
):
    print(f"[serve_workflow][DBG] Received async request from user '{username}'")
    user_overrides: Dict[str, Any] = _parse_overrides(overrides)
    print(f"[serve_workflow][DBG] User overrides: {json.dumps(user_overrides)}")
    filenames = _save_uploads(file50, file59)

    inputs = {
        "50": {"image": filenames["50"]},
        "59": {"image": filenames["59"]},
        **user_overrides
    }
    print(f"[serve_workflow][DBG] Inputs for ComfyUI: {json.dumps(inputs)}")

    job_id = uuid.uuid4().hex
    with JOB_LOCK:
        JOB_STORE[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": _now_iso(),
            "updated_at": _now_iso()
        }

    thread = threading.Thread(target=_run_job, args=(job_id, inputs, attempt_number), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str, username: str = Depends(check_auth)):
    with JOB_LOCK:
        job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, username: str = Depends(check_auth)):
    with JOB_LOCK:
        existed = JOB_STORE.pop(job_id, None) is not None
    if not existed:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "deleted": True}
