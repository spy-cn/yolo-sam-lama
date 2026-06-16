"""
OpenAI-compatible HTTP server for GLM-Edge-V (2B + 5B), with Bearer API key.
Optimized version with per-model locking, async logging, LRU model cache, and safety guards.
"""
import asyncio
import base64
import datetime
import io
import json
import logging
import os
import queue
import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
import torch
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class Settings(BaseSettings):
    default_model: str = "glm-edge-v-2b"
    api_key_file: str = "./.glm_api_key"
    api_keys_file: str = "./.glm_api_keys.json"
    access_log_file: str = "/var/log/glm-server-access.jsonl"
    access_log_enabled: bool = True
    preload_models: str = ""  # comma-separated, empty = all
    attn_impl: Optional[str] = None
    auto_mc: bool = True
    auto_mc_replace: bool = False
    max_image_size_mb: int = 10
    max_image_pixels: int = 20_000_000
    generation_timeout_s: int = 300
    log_batch_size: int = 100
    log_flush_interval_s: float = 0.1
    max_loaded_models: int = 3  # LRU cache size

    class Config:
        env_prefix = "GLM_"
        case_sensitive = False
        env_file = ".env"


settings = Settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("glm-server")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

GLM4_0_6B_PATH = str(BASE_DIR / "models/glm/glm4-edge-0.6b/glm4-edge-0.6b-2508v4")
GLM_2B_PATH = str(BASE_DIR / "models/glm/ZhipuAI/glm-edge-v-2b")
GLM4_5B_PATH = str(BASE_DIR / "models/glm/ZhipuAI/glm-edge-v-5b")

MODELS = {
    #"glm-edge-v-2b": {"path": GLM_2B_PATH, "vision": True},
    "glm-edge-v-5b": {"path": GLM4_5B_PATH, "vision": True},
    #"glm4-edge-0.6b": {"path": GLM4_0_6B_PATH, "vision": False},
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

_log_queue: queue.Queue = queue.Queue(maxsize=50000)
_log_writer_stop = threading.Event()


def _ensure_log_writable() -> str:
    parent = os.path.dirname(settings.access_log_file) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        with open(settings.access_log_file, "a"):
            pass
        return settings.access_log_file
    except (PermissionError, OSError):
        fallback = "/tmp/glm-server-access.jsonl"
        log.warning("Cannot write to %s, falling back to %s", settings.access_log_file, fallback)
        return fallback


_actual_log_file = _ensure_log_writable() if settings.access_log_enabled else None


def _log_writer_thread():
    while not _log_writer_stop.is_set():
        records = []
        try:
            while len(records) < settings.log_batch_size:
                records.append(_log_queue.get_nowait())
        except queue.Empty:
            pass
        if records and _actual_log_file:
            try:
                with open(_actual_log_file, "a", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            except Exception as e:
                log.warning("access log batch write failed: %s", e)
        _log_writer_stop.wait(settings.log_flush_interval_s)


if settings.access_log_enabled:
    threading.Thread(target=_log_writer_thread, daemon=True).start()


def _log_access(record: dict) -> None:
    if not settings.access_log_enabled:
        return
    record.setdefault("ts", datetime.datetime.utcnow().isoformat() + "Z")
    try:
        _log_queue.put_nowait(record)
    except queue.Full:
        log.warning("access log queue full, dropping record")


_gen_locks: dict[str, threading.Lock] = {}
_gen_locks_guard = threading.Lock()


def _get_gen_lock(model_name: str) -> threading.Lock:
    with _gen_locks_guard:
        if model_name not in _gen_locks:
            _gen_locks[model_name] = threading.Lock()
        return _gen_locks[model_name]


class LRULoadedModels:
    """Thread-safe LRU cache for loaded models with GPU memory awareness."""

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, name: str) -> Optional[dict]:
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]
            return None

    def put(self, name: str, bundle: dict) -> None:
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return
            # Evict LRU if at capacity
            while len(self._cache) >= self._max_size and self._cache:
                evict_name, evict_bundle = self._cache.popitem(last=False)
                log.info("[lru] evicting model=%s to make room for %s", evict_name, name)
                del evict_bundle["model"]
                del evict_bundle["tokenizer"]
                if evict_bundle.get("image_processor"):
                    del evict_bundle["image_processor"]
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            self._cache[name] = bundle

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()


_loaded_models = LRULoadedModels(settings.max_loaded_models)

_MC_OPTION_RE = re.compile(r"^\s*[\(\[]?(?P<option>[A-J])[\)\]\.\:]\s+\S", re.MULTILINE)
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
    letters = [m.group("option") for m in _MC_OPTION_RE.finditer(text)]
    return sorted(set(letters)) if len(set(letters)) >= 3 else None


def _maybe_augment_mc(messages: list[dict]) -> bool:
    if not settings.auto_mc or not messages:
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


_keys_lock = threading.RLock()
_KEY_POOL: dict[str, dict] = {}


def _persist_keys(entries: list[dict]) -> None:
    with open(settings.api_keys_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(settings.api_keys_file, 0o600)
    except Exception:
        pass


def _load_keys() -> dict[str, dict]:
    try:
        with open(settings.api_keys_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        entries = []
        # Migrate legacy single key
        try:
            with open(settings.api_key_file, "r", encoding="utf-8") as f:
                legacy = f.read().strip()
            if legacy:
                entries.append({
                    "key": legacy, "tenant": "legacy", "role": "user",
                    "active": True,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "note": "auto-migrated from legacy key file",
                })
        except FileNotFoundError:
            pass
        # Bootstrap admin key
        admin_key = "sk-admin-" + secrets.token_urlsafe(32)
        entries.append({
            "key": admin_key, "tenant": "ops", "role": "admin",
            "active": True,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "note": "bootstrap admin key",
        })
        try:
            _persist_keys(entries)
            log.info("bootstrapped key pool at %s (%d entries). Admin key: %s",
                     settings.api_keys_file, len(entries), admin_key)
        except Exception as e:
            log.warning("could not write %s: %s", settings.api_keys_file, e)

    pool: dict[str, dict] = {}
    for e in entries:
        if e.get("active", True) and e.get("key"):
            pool[e["key"]] = e
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
    import types

    def _shim(self, outputs, standardize_cache_format=False):
        cache_name = "past_key_values"
        cache = getattr(outputs, "past_key_values", None)
        if cache is None and hasattr(outputs, "mems"):
            cache = outputs.mems
        if cache is None and hasattr(outputs, "cache_params"):
            cache = outputs.cache_params
            cache_name = "cache_params"
        return cache_name, cache

    model._extract_past_from_model_output = types.MethodType(_shim, model)


def _check_gpu_memory(min_gb: float) -> bool:
    if DEVICE != "cuda":
        return True
    free_mem = (torch.cuda.get_device_properties(0).total_memory -
                torch.cuda.memory_allocated()) / 1024 ** 3
    return free_mem >= min_gb


def _load_model(name: str) -> dict:
    cfg = MODELS[name]
    path = cfg["path"]
    is_vision = cfg.get("vision", False)
    log.info("[load] %s from %s (vision=%s)", name, path, is_vision)

    # Pre-load GPU memory check (rough estimate: 2B≈5GB, 5B≈12GB, 0.6B≈2GB)
    mem_estimate = {"glm-edge-v-2b": 5.0, "glm-edge-v-5b": 12.0, "glm4-edge-0.6b": 2.0}
    if not _check_gpu_memory(mem_estimate.get(name, 4.0)):
        log.warning("[load] insufficient GPU memory for %s, attempting anyway", name)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    image_processor = None
    if is_vision:
        image_processor = AutoImageProcessor.from_pretrained(path, trust_remote_code=True)

    load_kwargs = dict(torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
    if settings.attn_impl:
        load_kwargs["attn_implementation"] = settings.attn_impl

    try:
        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs).eval()
        if settings.attn_impl:
            log.info("[load] %s using attn_implementation=%s", name, settings.attn_impl)
    except Exception as e:
        if "attn_implementation" in load_kwargs:
            log.warning("[load] %s with attn=%s failed (%s); retry default", name, settings.attn_impl, e)
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


def _get_or_load_model(name: str) -> dict:
    """Get model from LRU cache or load it."""
    bundle = _loaded_models.get(name)
    if bundle is not None:
        return bundle
    bundle = _load_model(name)
    _loaded_models.put(name, bundle)
    return bundle


_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_str = settings.preload_models or ",".join(MODELS.keys())
    preload = [n.strip() for n in preload_str.split(",") if n.strip()]
    for name in preload:
        if name in MODELS:
            try:
                _get_or_load_model(name)
            except Exception as e:
                log.error("[preload] failed to load %s: %s", name, e)
    log.info("ready. loaded=%s", _loaded_models.keys())
    yield
    _loaded_models.clear()
    _log_writer_stop.set()


app = FastAPI(title="GLM-Edge-V Multi-Model Server", lifespan=lifespan)
router = APIRouter()


class ImageUrlObj(BaseModel):
    url: str


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default_factory=lambda: settings.default_model)
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    tools: Optional[list[dict]] = None
    tool_choice: Any = None


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ARG_KV_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


def _parse_tool_calls(text: str, tool_names: Optional[list[str]] = None) -> tuple[Optional[str], list[dict]]:
    matches = list(_TOOL_CALL_RE.finditer(text))
    if matches:
        content_part = text[:matches[0].start()].strip()
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
    max_bytes = settings.max_image_size_mb * 1024 * 1024

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

    if len(raw) > max_bytes:
        raise ValueError(f"Image too large: {len(raw) / 1024 / 1024:.2f}MB > {settings.max_image_size_mb}MB")

    img = Image.open(io.BytesIO(raw))
    if img.width * img.height > settings.max_image_pixels:
        raise ValueError(f"Image resolution too high: {img.width}x{img.height} > {settings.max_image_pixels}px")

    return img.convert("RGB")


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


@router.get("/health")
def health():
    gpu = {}
    if DEVICE == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "mem_used_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
            "mem_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2),
        }
    model_status = {}
    for name, cfg in MODELS.items():
        model_status[name] = {
            "loaded": name in _loaded_models.keys(),
            "vision": cfg["vision"],
            "path": cfg["path"],
        }
    return {
        "status": "ok",
        "device": DEVICE,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "models": model_status,
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
                "loaded": m in _loaded_models.keys(),
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
        with open(_actual_log_file, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, n * 1024)
            f.seek(size - chunk)
            data = f.read().decode("utf-8", errors="replace")
        lines = [l for l in data.splitlines() if l.strip()]
    except (FileNotFoundError, TypeError):
        return {"file": _actual_log_file, "records": [], "count": 0}
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
    return {"file": _actual_log_file, "count": len(parsed), "records": parsed}


@router.get("/v1/admin/access-stats")
def admin_access_stats(window_minutes: int = 60, _=Depends(require_admin)):
    from collections import Counter
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(minutes=max(1, window_minutes))
    cutoff_str = cutoff.isoformat() + "Z"

    by_model: Counter = Counter()
    by_status: Counter = Counter()
    by_ip: Counter = Counter()
    by_tenant: Counter = Counter()
    tot_prompt = tot_completion = 0
    tot_latency = 0.0
    n_ok = n_err = n_total = 0

    try:
        with open(_actual_log_file, "r", encoding="utf-8", errors="replace") as f:
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
    except (FileNotFoundError, TypeError):
        return {"file": _actual_log_file, "total": 0}

    return {
        "file": _actual_log_file,
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
        with open(settings.api_keys_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        return {"file": settings.api_keys_file, "count": 0, "keys": []}
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
    return {"file": settings.api_keys_file, "count": len(out), "keys": out}


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
        with open(settings.api_keys_file, "r", encoding="utf-8") as f:
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
        with open(settings.api_keys_file, "r", encoding="utf-8") as f:
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


# =============================================================================
# Generation Logic
# =============================================================================
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

    lock = _get_gen_lock(req.model)
    t0 = time.time()
    with lock, torch.no_grad():
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

    if mc_aug and settings.auto_mc_replace:
        letter = _extract_mc_letter(text)
        if letter:
            text = letter

    tool_calls: list[dict] = []
    content_for_msg: Optional[str] = text
    if req.tools and not bundle.get("vision", False):
        tool_names = [t["function"]["name"] for t in req.tools
                      if isinstance(t, dict) and "function" in t]
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
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
    }


async def _stream_sse(bundle: dict, req: ChatCompletionRequest):
    """Async SSE generator using asyncio.to_thread for safe threading."""
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
    lock = _get_gen_lock(req.model)

    def _run_generate():
        with lock, torch.no_grad():
            model.generate(**gen_kwargs)

    # Run generation in thread pool via asyncio
    gen_task = asyncio.ensure_future(asyncio.to_thread(_run_generate))

    first = {
        "id": completion_id, "object": "chat.completion.chunk", "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

    total_chars = 0
    ttft = None
    t0 = time.time()

    try:
        while True:
            # Check streamer with timeout to allow cancellation
            try:
                chunk_text = await asyncio.wait_for(
                    asyncio.to_thread(streamer.__next__),
                    timeout=settings.generation_timeout_s
                )
            except StopIteration:
                break
            except asyncio.TimeoutError:
                log.warning("[stream] generation timeout after %ds for model=%s",
                            settings.generation_timeout_s, req.model)
                gen_task.cancel()
                error_payload = {
                    "id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                }
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

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
    except Exception as e:
        log.exception("[stream] unexpected error for model=%s", req.model)
        error_payload = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

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
async def chat_completions(req: ChatCompletionRequest, request: Request, auth: dict = Depends(require_user)):
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

    try:
        bundle = _get_or_load_model(req.model)
    except Exception as e:
        _finalize(500, error=f"load_failed: {e}")
        raise HTTPException(500, f"failed to load model {req.model}: {e}") from e

    if req.tools:
        if bundle.get("vision", False):
            _finalize(400, error="fc_on_vision")
            raise HTTPException(400, "function calling not supported on vision models")
        if req.stream:
            _finalize(400, error="fc_with_stream")
            raise HTTPException(400, "stream=true with tools not supported; use stream=false")

    if req.stream:
        gen = _stream_sse(bundle, req)

        async def _wrapped():
            tokens_out_chars = 0
            try:
                async for chunk in gen:
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
        # Run blocking generation in thread to avoid blocking event loop
        resp = await asyncio.to_thread(_generate_blocking, bundle, req)
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


app.include_router(router)
app.include_router(router, prefix="/aibox")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
