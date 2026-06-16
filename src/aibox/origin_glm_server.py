"""OpenAI-compatible HTTP server for GLM-Edge-V (2B + 5B), with Bearer API key."""
import base64
import datetime
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ---- config ----
# vision=True 表示是 VL 模型，需要 pixel_values；vision=False 是纯文本 LLM
BASE_DIR = Path(__file__).resolve().parent.parent.parent

GLM4_0_6B_PATH = str(BASE_DIR / "models/glm/glm4-edge-0.6b/glm4-edge-0.6b-2508v4")
GLM_2B_PATH = str(BASE_DIR / "models/glm/ZhipuAI/glm-edge-v-2b")
GLM4_5B_PATH = str(BASE_DIR / "models/glm/ZhipuAI/glm-edge-v-5b")

MODELS = {
    #"glm-edge-v-2b": {"path": GLM_2B_PATH, "vision": True},
    "glm-edge-v-5b": {"path": GLM4_5B_PATH, "vision": True},
    #"glm4-edge-0.6b": {"path": GLM4_0_6B_PATH, "vision": False},
}
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "glm-edge-v-2b")
API_KEY_FILE = os.environ.get("API_KEY_FILE", "./.glm_api_key")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("glm-server")

_state: dict[str, Any] = {"loaded": {}}  # model_name -> {tokenizer, image_processor, model}
_gen_lock = threading.Lock()

# ---- Access log (JSON Lines) ----
ACCESS_LOG_FILE = os.environ.get("ACCESS_LOG_FILE", "/var/log/glm-server-access.jsonl")
ACCESS_LOG_ENABLED = os.environ.get("ACCESS_LOG_ENABLED", "1") == "1"
_log_write_lock = threading.Lock()


def _ensure_log_writable() -> str:
    """Ensure the log file is writable; fall back to /tmp if not."""
    global ACCESS_LOG_FILE
    parent = os.path.dirname(ACCESS_LOG_FILE) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        # touch to verify write
        with open(ACCESS_LOG_FILE, "a"):
            pass
        return ACCESS_LOG_FILE
    except (PermissionError, OSError):
        ACCESS_LOG_FILE = "/tmp/glm-server-access.jsonl"
        return ACCESS_LOG_FILE


if ACCESS_LOG_ENABLED:
    _ensure_log_writable()


def _client_ip(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else None


def _log_access(record: dict) -> None:
    if not ACCESS_LOG_ENABLED:
        return
    record.setdefault("ts", datetime.datetime.utcnow().isoformat() + "Z")
    line = json.dumps(record, ensure_ascii=False, default=str)
    try:
        with _log_write_lock:
            with open(ACCESS_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        log.warning("access log write failed: %s", e)


# ---- Auto-MC (multiple-choice) optimization ----
AUTO_MC = os.environ.get("AUTO_MC", "1") == "1"
AUTO_MC_REPLACE = os.environ.get("AUTO_MC_REPLACE", "0") == "1"
_MC_OPTION_RE = re.compile(r"^\s*[\(\[]?([A-J])[\)\]\.\:]\s+\S", re.MULTILINE)
_MC_COT_SUFFIX = (
    "\n\nApproach this step by step: "
    "(1) Carefully observe the image. "
    "(2) Identify what the question is asking. "
    "(3) Reason through the options. "
    "(4) End your response with exactly one line in the format:\n X\n"
    "where X is the single letter of the correct option."
)
_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*[:：is]+\s*\(?([A-J])\)?", re.IGNORECASE)


def _detect_mc_letters(text: str) -> Optional[list[str]]:
    letters = _MC_OPTION_RE.findall(text)
    return sorted(set(letters)) if len(set(letters)) >= 3 else None


def _maybe_augment_mc(messages: list[dict]) -> bool:
    if not AUTO_MC or not messages:
        return False
    last = messages[-1]
    if last.get("role") != "user":
        return False
    content = last.get("content")

    def _augment_text(t: str) -> Optional[str]:
        if _MC_COT_SUFFIX in t:
            return None
        if _detect_mc_letters(t):
            return t + _MC_COT_SUFFIX
        return None

    if isinstance(content, str):
        new_t = _augment_text(content)
        if new_t:
            last["content"] = new_t
            return True
        return False
    if isinstance(content, list):
        for p in reversed(content):
            if isinstance(p, dict) and p.get("type") == "text":
                new_t = _augment_text(p.get("text", ""))
                if new_t:
                    p["text"] = new_t
                    return True
                break
    return False


def _extract_mc_letter(text: str) -> Optional[str]:
    m = _FINAL_ANSWER_RE.search(text)
    return m.group(1).upper() if m else None


# ---- API Key Pool ----
import secrets

API_KEYS_FILE = os.environ.get("API_KEYS_FILE", "./.glm_api_keys.json")
_keys_lock = threading.Lock()
_KEY_POOL: dict[str, dict] = {}

# /home/jdo/workspace/ford/models/ZhipuAI/glm4-edge-0.6b/glm4-edge-0.6b-2508v4

def _persist_keys(entries: list[dict]) -> None:
    with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(API_KEYS_FILE, 0o600)
    except Exception:
        pass


def _load_keys() -> dict[str, dict]:
    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        entries = []
        try:
            with open(API_KEY_FILE, "r", encoding="utf-8") as f:
                legacy = f.read().strip()
            if legacy:
                entries.append({
                    "key": legacy,
                    "tenant": "legacy",
                    "role": "user",
                    "active": True,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "note": "auto-migrated from /root/.glm_api_key",
                })
        except FileNotFoundError:
            pass
        admin_key = "sk-admin-" + secrets.token_urlsafe(32)
        entries.append({
            "key": admin_key,
            "tenant": "ops",
            "role": "admin",
            "active": True,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "note": "bootstrap admin key",
        })
        try:
            _persist_keys(entries)
            log.info("bootstrapped key pool at %s (%d entries). Admin key: %s",
                     API_KEYS_FILE, len(entries), admin_key)
        except Exception as e:
            log.warning("could not write %s: %s", API_KEYS_FILE, e)

    pool: dict[str, dict] = {}
    for e in entries:
        if not e.get("active", True):
            continue
        k = e.get("key")
        if k:
            pool[k] = e
    return pool


_KEY_POOL = _load_keys()
bearer = HTTPBearer(auto_error=False)


def _resolve_key(creds: Optional[HTTPAuthorizationCredentials]) -> dict:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="invalid api key")
    with _keys_lock:
        meta = _KEY_POOL.get(creds.credentials)
    if not meta:
        raise HTTPException(status_code=401, detail="invalid api key")
    return meta


def require_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> dict:
    return _resolve_key(creds)


def require_admin(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> dict:
    meta = _resolve_key(creds)
    if meta.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return meta


def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 12:
        return k[:2] + "***"
    return k[:8] + "..." + k[-4:]


def _patch_legacy_extract_past(model) -> None:
    if hasattr(model, "_extract_past_from_model_output"):
        return

    def _shim(self, outputs, standardize_cache_format=False):
        cache_name = "past_key_values"
        cache = getattr(outputs, "past_key_values", None)
        if cache is None and hasattr(outputs, "mems"):
            cache = outputs.mems
        if cache is None and hasattr(outputs, "cache_params"):
            cache = outputs.cache_params
            cache_name = "cache_params"
        return cache_name, cache

    import types
    model._extract_past_from_model_output = types.MethodType(_shim, model)


def _load_model(name: str) -> dict:
    cfg = MODELS[name]
    path = cfg["path"]
    is_vision = cfg.get("vision", False)
    log.info("[load] %s from %s (vision=%s)", name, path, is_vision)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    image_processor = None
    if is_vision:
        image_processor = AutoImageProcessor.from_pretrained(path, trust_remote_code=True)
    load_kwargs = dict(dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
    load_kwargs["dtype"] = DTYPE
    attn_impl = os.environ.get("ATTN_IMPL")
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs).eval()
        if attn_impl:
            log.info("[load] %s using attn_implementation=%s", name, attn_impl)
    except Exception as e:
        if "attn_implementation" in load_kwargs:
            log.warning("[load] %s with attn=%s failed (%s); retry default", name, attn_impl, e)
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs).eval()
        else:
            raise
    _patch_legacy_extract_past(model)
    mem = torch.cuda.memory_allocated() / 1024 ** 3 if DEVICE == "cuda" else 0
    log.info("[load] %s done in %.2fs total_gpu=%.2fGB", name, time.time() - t0, mem)
    return {
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "model": model,
        "vision": is_vision,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload = os.environ.get("PRELOAD", ",".join(MODELS.keys())).split(",")
    for name in preload:
        name = name.strip()
        if name and name in MODELS:
            _state["loaded"][name] = _load_model(name)
    log.info("ready. loaded=%s", list(_state["loaded"].keys()))
    yield
    _state["loaded"].clear()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="GLM-Edge-V Multi-Model Server", lifespan=lifespan)
router = APIRouter()


# ---- schemas ----
class ImageUrlObj(BaseModel):
    url: str


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    tools: Optional[list[dict]] = None
    tool_choice: Any = None


# ---- Function-calling parsing ----
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ARG_KV_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


def _parse_tool_calls(text: str, tool_names: Optional[list[str]] = None) -> tuple[Optional[str], list[dict]]:
    matches = list(_TOOL_CALL_RE.finditer(text))
    if matches:
        content_part = text[: matches[0].start()].strip()
        calls = []
        for m in matches:
            body = m.group(1).strip()
            lines = body.split("\n", 1)
            fn_name = lines[0].strip()
            rest = lines[1] if len(lines) > 1 else ""
            args: dict[str, Any] = {}
            for k, v in _ARG_KV_RE.findall(rest):
                k, v = k.strip(), v.strip()
                try:
                    args[k] = json.loads(v)
                except Exception:
                    args[k] = v
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": fn_name, "arguments": json.dumps(args, ensure_ascii=False)},
            })
        return (content_part if content_part else None), calls

    if not tool_names:
        return text, []
    lines = [l.strip() for l in text.split("\n")]
    fn_idx = None
    fn_name = None
    for i, l in enumerate(lines):
        if l in tool_names:
            fn_idx = i
            fn_name = l
            break
    if fn_idx is None:
        return text, []
    content_part = "\n".join(lines[:fn_idx]).strip()
    arg_lines = [l for l in lines[fn_idx + 1:] if l]
    args: dict[str, Any] = {}
    for i in range(0, len(arg_lines) - 1, 2):
        k, v = arg_lines[i], arg_lines[i + 1]
        try:
            args[k] = json.loads(v)
        except Exception:
            args[k] = v
    call = {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": fn_name, "arguments": json.dumps(args, ensure_ascii=False)},
    }
    return (content_part if content_part else None), [call]


def decode_image(url_or_data: str) -> Image.Image:
    if url_or_data.startswith("data:"):
        _, b64 = url_or_data.split(",", 1)
        raw = base64.b64decode(b64)
    elif url_or_data.startswith(("http://", "https://")):
        import urllib.request
        req = urllib.request.Request(url_or_data, headers={"User-Agent": "glm-server/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
    else:
        raw = base64.b64decode(url_or_data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _concat_images_vertical(imgs: list[Image.Image]) -> Image.Image:
    if len(imgs) == 1:
        return imgs[0]
    target_w = max(im.width for im in imgs)
    scaled = []
    total_h = 0
    for im in imgs:
        ratio = target_w / im.width
        new_h = int(im.height * ratio)
        scaled.append(im.resize((target_w, new_h), Image.LANCZOS))
        total_h += new_h
    out = Image.new("RGB", (target_w, total_h), "white")
    y = 0
    for im in scaled:
        out.paste(im, (0, y))
        y += im.height
    return out


def extract_images_and_text(messages: list[ChatMessage]) -> tuple[list[Image.Image], list[dict]]:
    images: list[Image.Image] = []
    out: list[dict] = []
    for msg in messages:
        content = msg.content
        if content is None:
            out.append({"role": msg.role, "content": ""})
            continue
        if isinstance(content, str):
            out.append({"role": msg.role, "content": content})
            continue
        parts: list[dict] = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif t == "image_url":
                    iu = p["image_url"]
                    url = iu["url"] if isinstance(iu, dict) else iu.url
                    images.append(decode_image(url))
                    if not any(pp.get("type") == "image" for pp in parts):
                        parts.append({"type": "image"})
            else:
                if p.type == "text":
                    parts.append({"type": "text", "text": p.text})
                elif p.type == "image_url":
                    images.append(decode_image(p.image_url.url))
                    if not any(pp.get("type") == "image" for pp in parts):
                        parts.append({"type": "image"})
        out.append({"role": msg.role, "content": parts})
    return images, out


# ---- endpoints ----
@router.get("/health")
def health():
    gpu = {}
    if DEVICE == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "mem_used_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
            "mem_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2),
        }
    return {
        "status": "ok",
        "device": DEVICE,
        "loaded_models": list(_state["loaded"].keys()),
        "configured_models": list(MODELS.keys()),
        "gpu": gpu,
    }


@router.get("/v1/models")
def list_models(_=Depends(require_user)):
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "owned_by": "zhipu",
                "loaded": m in _state["loaded"],
            }
            for m in MODELS
        ],
    }


@router.get("/v1/admin/access-log")
def admin_access_log(
    n: int = 50,
    model: Optional[str] = None,
    ip: Optional[str] = None,
    status: Optional[int] = None,
    tenant: Optional[str] = None,
    _=Depends(require_admin),
):
    n = max(1, min(int(n), 1000))
    try:
        with open(ACCESS_LOG_FILE, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, n * 1024)
            f.seek(size - chunk)
            data = f.read().decode("utf-8", errors="replace")
        lines = [l for l in data.splitlines() if l.strip()]
    except FileNotFoundError:
        return {"file": ACCESS_LOG_FILE, "records": [], "count": 0}
    parsed = []
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if model and rec.get("model") != model:
            continue
        if ip and rec.get("ip") != ip:
            continue
        if status is not None and rec.get("status") != status:
            continue
        if tenant and rec.get("tenant") != tenant:
            continue
        parsed.append(rec)
        if len(parsed) >= n:
            break
    return {"file": ACCESS_LOG_FILE, "count": len(parsed), "records": parsed}


@router.get("/v1/admin/access-stats")
def admin_access_stats(window_minutes: int = 60, _=Depends(require_admin)):
    from collections import Counter
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(minutes=max(1, window_minutes))
    cutoff_str = cutoff.isoformat() + "Z"

    by_model: Counter = Counter()
    by_status: Counter = Counter()
    by_ip: Counter = Counter()
    by_tenant: Counter = Counter()
    tot_prompt = 0
    tot_completion = 0
    tot_latency = 0.0
    n_ok = 0
    n_err = 0
    n_total = 0
    try:
        with open(ACCESS_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ts = rec.get("ts", "")
                if ts and ts < cutoff_str:
                    continue
                n_total += 1
                by_model[rec.get("model") or "?"] += 1
                by_status[str(rec.get("status") or "?")] += 1
                by_ip[rec.get("ip") or "?"] += 1
                by_tenant[rec.get("tenant") or "?"] += 1
                tot_prompt += rec.get("prompt_tokens") or 0
                tot_completion += rec.get("completion_tokens") or 0
                tot_latency += rec.get("latency_ms") or 0
                if rec.get("status") == 200:
                    n_ok += 1
                else:
                    n_err += 1
    except FileNotFoundError:
        return {"file": ACCESS_LOG_FILE, "total": 0}
    return {
        "file": ACCESS_LOG_FILE,
        "window_minutes": window_minutes,
        "total": n_total,
        "ok": n_ok,
        "error": n_err,
        "by_model": dict(by_model),
        "by_status": dict(by_status),
        "by_tenant": dict(by_tenant),
        "top_ips": dict(by_ip.most_common(10)),
        "total_prompt_tokens": tot_prompt,
        "total_completion_tokens": tot_completion,
        "avg_latency_ms": round(tot_latency / n_total, 1) if n_total else 0,
    }


@router.get("/v1/admin/keys")
def admin_list_keys(_=Depends(require_admin)):
    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        return {"file": API_KEYS_FILE, "count": 0, "keys": []}
    out = []
    for e in entries:
        out.append({
            "key_masked": _mask_key(e.get("key", "")),
            "tenant": e.get("tenant"),
            "role": e.get("role", "user"),
            "active": e.get("active", True),
            "created_at": e.get("created_at"),
            "note": e.get("note"),
        })
    return {"file": API_KEYS_FILE, "count": len(out), "keys": out}


class KeyCreate(BaseModel):
    tenant: str
    role: str = "user"
    note: Optional[str] = None
    key: Optional[str] = None


@router.post("/v1/admin/keys")
def admin_create_key(body: KeyCreate, _=Depends(require_admin)):
    if body.role not in ("user", "admin"):
        raise HTTPException(400, "role must be 'user' or 'admin'")
    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        entries = []
    new_key = body.key or ("sk-" + body.tenant.replace("_", "-") + "-" + secrets.token_urlsafe(24))
    if any(e.get("key") == new_key for e in entries):
        raise HTTPException(409, "key already exists")
    entry = {
        "key": new_key,
        "tenant": body.tenant,
        "role": body.role,
        "active": True,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if body.note:
        entry["note"] = body.note
    entries.append(entry)
    _persist_keys(entries)
    global _KEY_POOL
    with _keys_lock:
        _KEY_POOL = _load_keys()
    log.info("admin: created key for tenant=%s role=%s", body.tenant, body.role)
    return {"created": True, "key": new_key, "tenant": body.tenant, "role": body.role}


class KeyUpdate(BaseModel):
    tenant: str
    active: bool


@router.post("/v1/admin/keys/set-active")
def admin_set_key_active(body: KeyUpdate, _=Depends(require_admin)):
    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        raise HTTPException(404, "no keys file")
    updated = 0
    for e in entries:
        if e.get("tenant") == body.tenant:
            e["active"] = body.active
            updated += 1
    if updated == 0:
        raise HTTPException(404, f"no keys for tenant={body.tenant}")
    _persist_keys(entries)
    global _KEY_POOL
    with _keys_lock:
        _KEY_POOL = _load_keys()
    log.info("admin: set active=%s for tenant=%s (%d keys)", body.active, body.tenant, updated)
    return {"updated": updated, "tenant": body.tenant, "active": body.active}


@router.post("/v1/admin/keys/reload")
def admin_reload_keys(_=Depends(require_admin)):
    global _KEY_POOL
    with _keys_lock:
        _KEY_POOL = _load_keys()
    return {"reloaded": True, "active_keys": len(_KEY_POOL)}


def _build_gen_kwargs(bundle: dict, req: ChatCompletionRequest) -> tuple[dict, int, bool]:
    tokenizer = bundle["tokenizer"]
    image_processor = bundle["image_processor"]
    is_vision = bundle.get("vision", False)
    try:
        images, msgs = extract_images_and_text(req.messages)
    except Exception as e:
        raise HTTPException(400, f"failed to parse messages: {e}") from e

    mc_aug = _maybe_augment_mc(msgs)
    if mc_aug:
        log.info("[mc] auto-augmented prompt with CoT for model=%s", req.model)

    if not is_vision:
        flat = []
        for orig, m in zip(req.messages, msgs):
            c = m["content"]
            if isinstance(c, list):
                texts = [p["text"] for p in c if p.get("type") == "text"]
                if images and not texts:
                    texts = ["[image input ignored: text-only model]"]
                content = "\n".join(texts)
            else:
                content = c
            entry: dict[str, Any] = {"role": m["role"], "content": content}
            if orig.tool_calls:
                entry["tool_calls"] = orig.tool_calls
            if orig.tool_call_id:
                entry["tool_call_id"] = orig.tool_call_id
            if orig.name:
                entry["name"] = orig.name
            flat.append(entry)
        msgs = flat

    use_tools = (not is_vision) and bool(req.tools) and (req.tool_choice != "none")
    template_kwargs: dict[str, Any] = dict(
        add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors="pt",
    )
    if use_tools:
        template_kwargs["tools"] = req.tools

    inputs = tokenizer.apply_chat_template(msgs, **template_kwargs).to(DEVICE)
    gen_kwargs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    if is_vision:
        if images:
            merged = _concat_images_vertical(images) if len(images) > 1 else images[0]
            gen_kwargs["pixel_values"] = torch.tensor(image_processor(merged).pixel_values).to(DEVICE)
        else:
            gen_kwargs["pixel_values"] = torch.zeros([1, 1, 1, 3, 672, 672], dtype=DTYPE, device=DEVICE)

    n_in = int(inputs["input_ids"].shape[1])
    return gen_kwargs, n_in, mc_aug


def _generate_blocking(bundle: dict, req: ChatCompletionRequest) -> dict:
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    gen_kwargs, n_in, mc_aug = _build_gen_kwargs(bundle, req)
    effective_max = req.max_tokens
    if mc_aug and effective_max < 192:
        effective_max = 256
    t0 = time.time()
    with _gen_lock, torch.no_grad():
        out = model.generate(
            **gen_kwargs,
            max_new_tokens=effective_max,
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 1e-5),
            top_p=req.top_p,
        )
    elapsed = time.time() - t0
    new_tokens = out[0, n_in:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    n_out = int(new_tokens.shape[0])
    if mc_aug and AUTO_MC_REPLACE:
        letter = _extract_mc_letter(text)
        if letter:
            text = letter

    tool_calls: list[dict] = []
    content_for_msg: Optional[str] = text
    if req.tools and not bundle.get("vision", False):
        tool_names = [t["function"]["name"] for t in req.tools if isinstance(t, dict) and "function" in t]
        content_part, tool_calls = _parse_tool_calls(text, tool_names=tool_names)
        if tool_calls:
            content_for_msg = content_part

    log.info("[gen] model=%s in=%d out=%d %.2fs (%.1f tok/s)%s%s",
             req.model, n_in, n_out, elapsed, n_out / max(elapsed, 1e-3),
             " [mc-aug]" if mc_aug else "",
             f" [tool_calls={len(tool_calls)}]" if tool_calls else "")

    message: dict[str, Any] = {"role": "assistant", "content": content_for_msg}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish_reason = "tool_calls" if tool_calls else "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
    }


def _stream_sse(bundle: dict, req: ChatCompletionRequest):
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    gen_kwargs, n_in, mc_aug = _build_gen_kwargs(bundle, req)
    effective_max = req.max_tokens
    if mc_aug and effective_max < 192:
        effective_max = 256
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs.update(
        max_new_tokens=effective_max,
        do_sample=req.temperature > 0,
        temperature=max(req.temperature, 1e-5),
        top_p=req.top_p,
        streamer=streamer,
    )
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def gen_thread():
        with _gen_lock, torch.no_grad():
            try:
                model.generate(**gen_kwargs)
            except Exception as ex:
                log.exception("stream gen failure")
                streamer.end()
                raise ex

    t0 = time.time()
    t = threading.Thread(target=gen_thread, daemon=True)
    t.start()

    first = {
        "id": completion_id, "object": "chat.completion.chunk", "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

    total_chars = 0
    ttft = None
    for chunk_text in streamer:
        if not chunk_text:
            continue
        if ttft is None:
            ttft = time.time() - t0
        total_chars += len(chunk_text)
        payload = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    last = {
        "id": completion_id, "object": "chat.completion.chunk", "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
    elapsed = time.time() - t0
    log.info("[stream] model=%s in=%d chars=%d ttft=%.2fs total=%.2fs",
             req.model, n_in, total_chars, ttft or 0, elapsed)


@router.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, request: Request, auth: dict = Depends(require_user)):
    t_start = time.time()
    req_id = uuid.uuid4().hex[:16]
    client_ip = _client_ip(request)
    tenant = auth.get("tenant", "?")

    has_image = False
    text_chars = 0
    n_msgs = len(req.messages) if req.messages else 0
    try:
        for m in req.messages or []:
            c = m.content
            if isinstance(c, list):
                for p in c:
                    if isinstance(p, dict):
                        if p.get("type") == "image_url":
                            has_image = True
                        elif p.get("type") == "text":
                            text_chars += len(p.get("text", ""))
                    elif hasattr(p, "type"):
                        if p.type == "image_url":
                            has_image = True
                        elif p.type == "text":
                            text_chars += len(p.text or "")
            elif isinstance(c, str):
                text_chars += len(c)
    except Exception:
        pass

    base_record: dict[str, Any] = {
        "req_id": req_id,
        "ip": client_ip,
        "tenant": tenant,
        "model": req.model,
        "n_messages": n_msgs,
        "has_image": has_image,
        "has_tools": bool(req.tools),
        "stream": req.stream,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "prompt_chars": text_chars,
    }

    def _finalize(status: int, **extra: Any) -> None:
        extra.setdefault("latency_ms", round((time.time() - t_start) * 1000, 1))
        _log_access({**base_record, "status": status, **extra})

    if req.model not in MODELS:
        _finalize(404, error="unknown_model")
        raise HTTPException(404, f"unknown model: {req.model}; available={list(MODELS.keys())}")
    if req.model not in _state["loaded"]:
        try:
            _state["loaded"][req.model] = _load_model(req.model)
        except Exception as e:
            _finalize(500, error=f"load_failed: {e}")
            raise HTTPException(500, f"failed to load model {req.model}: {e}") from e

    bundle = _state["loaded"][req.model]

    if req.tools:
        if bundle.get("vision", False):
            _finalize(400, error="fc_on_vision")
            raise HTTPException(400, "function calling not supported on vision models")
        if req.stream:
            _finalize(400, error="fc_with_stream")
            raise HTTPException(400, "stream=true with tools not supported; use stream=false")

    if req.stream:
        gen = _stream_sse(bundle, req)

        def _wrapped():
            tokens_out_chars = 0
            try:
                for chunk in gen:
                    if '"content":' in chunk:
                        tokens_out_chars += chunk.count('"content":')
                    yield chunk
                _finalize(200, mode="stream", out_chars=tokens_out_chars)
            except Exception as e:
                _finalize(500, mode="stream", error=str(e)[:200])
                raise

        return StreamingResponse(
            _wrapped(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        resp = _generate_blocking(bundle, req)
        usage = resp.get("usage", {})
        choice = resp.get("choices", [{}])[0]
        finish = choice.get("finish_reason")
        msg = choice.get("message", {})
        tc_count = len(msg.get("tool_calls") or []) if msg else 0
        _finalize(
            200, mode="blocking",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            finish_reason=finish,
            tool_calls=tc_count,
        )
        return resp
    except HTTPException as he:
        _finalize(he.status_code, error=str(he.detail)[:200])
        raise
    except Exception as e:
        _finalize(500, error=str(e)[:200])
        raise

# ====== 路由挂载逻辑修改部分 ======
# 1. 挂载到根路径：保证你本地在服务器里 curl 127.0.0.1:8000 依然畅通无阻
app.include_router(router)

# 2. 挂载到带前缀路径：适配运维不抹除前缀的请求
app.include_router(router, prefix="/aibox")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
