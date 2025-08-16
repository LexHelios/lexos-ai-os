from fastapi import FastAPI, APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, AsyncGenerator
import uuid
from datetime import datetime, timedelta
import httpx
import time
import csv
from io import StringIO
from collections import deque, defaultdict
from statistics import median
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Emergent Integrations
try:
    from emergentintegrations import EmergentLLM  # type: ignore
except Exception:
    EmergentLLM = None

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# App init
app = FastAPI()
api_router = APIRouter(prefix="/api")

APP_VERSION = "1.4.0"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

LOG_SAMPLES = os.environ.get("LOG_SAMPLES", "false").lower() == "true"

# Prometheus metrics
REQ_COUNTER = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "path"]) 
LLM_REQUESTS = Counter("llm_requests_total", "LLM requests", ["provider", "model"])
LLM_FAILURES = Counter("llm_failures_total", "LLM failures", ["provider", "model", "code"])
LLM_LATENCY = Histogram("llm_latency_seconds", "LLM latency", ["provider", "model"]) 

@app.middleware("http")
async def metrics_logging_middleware(request: Request, call_next):
    start = time.time()
    trace_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    path = request.url.path
    method = request.method
    try:
        response: Response = await call_next(request)
        return response
    finally:
        duration = time.time() - start
        try:
            REQ_COUNTER.labels(method=method, path=path, status=str(response.status_code if 'response' in locals() else 500)).inc()
            REQ_LATENCY.labels(method=method, path=path).observe(duration)
        except Exception:
            pass
        LOGGER.info(
            "trace_id=%s method=%s path=%s status=%s latency_ms=%.2f",
            trace_id, method, path, (response.status_code if 'response' in locals() else 'NA'), duration * 1000
        )

# Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "ignore"

class StatusCheckCreate(BaseModel):
    client_name: str

class HealthResponse(BaseModel):
    status: str
    mongo: str
    time: datetime

class VersionResponse(BaseModel):
    version: str
    time: datetime

class StatusCountResponse(BaseModel):
    total: int
    distinct_clients: int

class PurgeRequest(BaseModel):
    client_name: Optional[str] = None
    older_than_hours: Optional[int] = Field(default=None, ge=1)

# AI schemas
class AIChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class AIChatRequest(BaseModel):
    messages: List[AIChatMessage]
    model: Optional[str] = Field(default="default")
    temperature: Optional[float] = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=500)
    provider: Optional[str] = Field(default=None)
    image_url: Optional[str] = Field(default=None)
    image_b64: Optional[str] = Field(default=None)
    image_mime: Optional[str] = Field(default=None)

class AISummaryRequest(BaseModel):
    hours: Optional[int] = Field(default=24, ge=1, le=168)
    limit: Optional[int] = Field(default=200, ge=1, le=2000)
    model: Optional[str] = Field(default="default")
    temperature: Optional[float] = Field(default=0.3)

class AIInsightsRequest(BaseModel):
    hours: Optional[int] = Field(default=24, ge=1, le=720)
    limit: Optional[int] = Field(default=500, ge=1, le=5000)

# Helpers
async def init_indexes():
    try:
        await db.status_checks.create_index("client_name")
        await db.status_checks.create_index("timestamp")
        LOGGER.info("MongoDB indexes ensured")
    except Exception as e:
        LOGGER.warning(f"Index creation failed: {e}")

def strip_mongo_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    doc.pop("_id", None)
    return doc

# Telemetry store
class Telemetry:
    def __init__(self):
        self.latencies = defaultdict(lambda: deque(maxlen=500))  # key: (provider, model)
        self.errors = deque(maxlen=200)  # list of {ts, provider, code, http_status, message}

    def record_latency(self, provider: str, model: str, seconds: float):
        self.latencies[(provider, model)].append(seconds)

    def record_error(self, provider: str, model: str, code: str, http_status: int, message: str):
        self.errors.append({
            "ts": datetime.utcnow().isoformat(),
            "provider": provider,
            "model": model,
            "code": code,
            "http_status": http_status,
            "message": message[:500],
        })

    def snapshot(self):
        # Compute simple stats per provider across models
        prov_stats: Dict[str, Dict[str, Any]] = {}
        for (provider, model), vals in self.latencies.items():
            if not vals:
                continue
            arr = list(vals)
            arr_sorted = sorted(arr)
            p50 = arr_sorted[int(0.5 * (len(arr_sorted)-1))]
            p95 = arr_sorted[int(0.95 * (len(arr_sorted)-1))]
            stat = prov_stats.setdefault(provider, {"models": {}, "p95": 0.0, "p50": 0.0})
            stat["models"][model] = {"count": len(arr), "p50": p50, "p95": p95}
        # Aggregate provider-wide
        for provider, s in prov_stats.items():
            vals = []
            for m in s["models"].values():
                vals.extend([m["p95"]])
            s["p95"] = max(vals) if vals else 0.0
            vals = []
            for m in s["models"].values():
                vals.extend([m["p50"]])
            s["p50"] = median(vals) if vals else 0.0
        return {
            "providers": prov_stats,
            "recent_errors": list(self.errors)[-20:],
        }

telemetry = Telemetry()

# Model aliasing
MODEL_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "gpt-4o-mini": {
        "openrouter": "openai/gpt-4o-mini",
        "together": None,
        "emergent": "gpt-4o-mini",
        "local": "gpt-4o-mini",
    },
    "claude-3.7-sonnet": {
        "openrouter": "anthropic/claude-3.7-sonnet",
        "together": None,
        "emergent": "claude-3.7-sonnet",
        "local": "claude-3.7-sonnet",
    },
    "llama-3.1-70b": {
        "openrouter": "meta-llama/llama-3.1-70b-instruct",
        "together": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        "emergent": "llama-3.1-70b",
        "local": "llama3.1:70b",
    },
}

# LLM Orchestrator with breaker and retries
class LLMOrchestrator:
    def __init__(self):
        self.local_base = os.environ.get("LLM_LOCAL_BASE_URL")
        self.together_base = os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
        self.together_key = os.environ.get("TOGETHER_API_KEY")
        self.openrouter_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "30"))
        self.enable_fallback = os.environ.get("ENABLE_LLM_FALLBACK", "true").lower() == "true"
        self.emergent_key = os.environ.get("EMERGENT_LLM_KEY")
        self.emergent_client = EmergentLLM(api_key=self.emergent_key) if (EmergentLLM and self.emergent_key) else None
        self.order = ["local", "together", "openrouter", "emergent"]
        now = 0
        self.breakers: Dict[str, Dict[str, Any]] = {p: {"open": False, "until": now, "failures": 0, "last_error": None} for p in self.order}

    # Breaker helpers
    def _breaker_allows(self, p: str) -> bool:
        b = self.breakers[p]
        if not b["open"]:
            return True
        if time.time() >= b["until"]:
            b["open"] = False
            b["failures"] = 0
            b["last_error"] = None
            return True
        return False

    def _breaker_trip(self, p: str, cool: int = 60, err: Optional[str] = None):
        b = self.breakers[p]
        b["failures"] += 1
        b["last_error"] = (err or "")[:300]
        if b["failures"] >= 3:
            b["open"] = True
            b["until"] = time.time() + cool

    # Error normalization
    def _err(self, provider: str, http_status: int, code: str, message: str, retriable: bool = True):
        return {
            "provider": provider,
            "http_status": http_status,
            "code": code,
            "message": message,
            "retriable": retriable,
        }

    # Model resolving
    def _resolve_model(self, provider: str, friendly: str) -> str:
        m = MODEL_MAP.get(friendly, {})
        return m.get(provider) or friendly

    # Message normalization to OpenAI-style blocks
    def _to_provider_messages(self, messages: List[Dict[str, Any]], image_url: Optional[str] = None, image_b64: Optional[str] = None, image_mime: Optional[str] = None):
        out = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            content = [{"type": "text", "text": text}]
            out.append({"role": role, "content": content})
        # Append image to last user message if present
        if (image_url or image_b64) and out:
            if out[-1]["role"] != "user":
                out.append({"role": "user", "content": []})
            if image_url:
                out[-1]["content"].append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                out[-1]["content"].append({"type": "input_image", "image": {"data": image_b64, "mime_type": image_mime or "image/png"}})
        return out

    async def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()

    async def _local_chat(self, payload_openai: Dict[str, Any], stream: bool = False) -> Any:
        if not self.local_base:
            raise RuntimeError("Local not configured")
        if not stream:
            try:
                data = await self._post_json(f"{self.local_base}/v1/chat/completions", payload_openai)
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception:
                # try Ollama-compatible
                ollama_payload = {
                    "model": payload_openai.get("model", "llama3"),
                    "messages": payload_openai.get("messages", []),
                    "stream": False,
                }
                data = await self._post_json(f"{self.local_base}/api/chat", ollama_payload)
                return data.get("message", {}).get("content", "")
        else:
            # Stream via OpenAI SSE style; fallback to single chunk
            async def gen() -> AsyncGenerator[str, None]:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        async with client.stream("POST", f"{self.local_base}/v1/chat/completions", json={**payload_openai, "stream": True}) as resp:
                            async for line in resp.aiter_lines():
                                if not line or not line.startswith("data: "):
                                    continue
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    j = httpx.Response(200, content=data).json()
                                except Exception:
                                    continue
                                try:
                                    delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
                                    if delta:
                                        yield delta
                                except Exception:
                                    continue
                except Exception:
                    # fallback non-stream
                    text = await self._local_chat(payload_openai, stream=False)
                    if text:
                        yield text
                yield ""
            return gen()

    async def _together_chat(self, payload_openai: Dict[str, Any], stream: bool = False) -> Any:
        if not self.together_key:
            raise RuntimeError("Together not configured")
        headers = {"Authorization": f"Bearer {self.together_key}", "Content-Type": "application/json"}
        if not stream:
            data = await self._post_json(f"{self.together_base}/chat/completions", payload_openai, headers)
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            async def gen() -> AsyncGenerator[str, None]:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        async with client.stream("POST", f"{self.together_base}/chat/completions", json={**payload_openai, "stream": True}, headers=headers) as resp:
                            async for line in resp.aiter_lines():
                                if not line or not line.startswith("data: "):
                                    continue
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    j = httpx.Response(200, content=data).json()
                                except Exception:
                                    continue
                                delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
                                if delta:
                                    yield delta
                except Exception:
                    yield ""
                yield ""
            return gen()

    async def _openrouter_chat(self, payload_openai: Dict[str, Any], stream: bool = False) -> Any:
        if not self.openrouter_key:
            raise RuntimeError("OpenRouter not configured")
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "Neo System"),
        }
        if not stream:
            data = await self._post_json(f"{self.openrouter_base}/chat/completions", payload_openai, headers)
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            async def gen() -> AsyncGenerator[str, None]:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        async with client.stream("POST", f"{self.openrouter_base}/chat/completions", json={**payload_openai, "stream": True}, headers=headers) as resp:
                            async for line in resp.aiter_lines():
                                if not line or not line.startswith("data: "):
                                    continue
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    j = httpx.Response(200, content=data).json()
                                except Exception:
                                    continue
                                delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
                                if delta:
                                    yield delta
                except Exception:
                    yield ""
                yield ""
            return gen()

    async def _emergent_chat(self, model: str, messages: List[Dict[str, Any]], stream: bool = False) -> Any:
        if not self.emergent_client:
            raise RuntimeError("Emergent not configured")
        if not stream:
            resp = self.emergent_client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content
        else:
            # Fallback: stream not guaranteed; emit single chunk
            resp = self.emergent_client.chat.completions.create(model=model, messages=messages)
            async def gen() -> AsyncGenerator[str, None]:
                yield resp.choices[0].message.content
                yield ""
            return gen()

    async def _call_with_retry(self, func, provider: str, model: str, *args, stream: bool = False, **kwargs):
        delay = 0.25
        max_retries = 2
        for attempt in range(max_retries + 1):
            start = time.time()
            try:
                result = await func(*args, stream=stream, **kwargs)
                LLM_LATENCY.labels(provider=provider, model=model).observe(time.time() - start)
                return result
            except httpx.TimeoutException as e:
                telemetry.record_error(provider, model, "timeout", 504, str(e))
                LLM_FAILURES.labels(provider=provider, model=model, code="timeout").inc()
                self._breaker_trip(provider, err="timeout")
                if attempt < max_retries:
                    await asyncio_sleep(delay)
                    delay *= 2
                    continue
                raise
            except Exception as e:
                telemetry.record_error(provider, model, "upstream_error", 502, str(e))
                LLM_FAILURES.labels(provider=provider, model=model, code="upstream_error").inc()
                self._breaker_trip(provider, err=str(e))
                if attempt < max_retries:
                    await asyncio_sleep(delay)
                    delay *= 2
                    continue
                raise

    async def chat(self, messages: List[Dict[str, Any]], model: str = "default", temperature: Optional[float] = None, max_tokens: Optional[int] = None, provider: Optional[str] = None, image_url: Optional[str] = None, image_b64: Optional[str] = None, image_mime: Optional[str] = None) -> Dict[str, Any]:
        # Build payload
        order_default = ["local", "together", "openrouter", "emergent"]
        order = []
        if provider in order_default:
            order = [provider] + [p for p in order_default if p != provider] if self.enable_fallback else [provider]
        else:
            order = order_default

        errors = []
        for p in order:
            if not self._breaker_allows(p):
                continue
            resolved_model = self._resolve_model(p, model)
            norm_messages = self._to_provider_messages(messages, image_url=image_url, image_b64=image_b64, image_mime=image_mime)
            payload = {"model": resolved_model, "messages": norm_messages}
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            LLM_REQUESTS.labels(provider=p, model=resolved_model).inc()
            try:
                if p == "local":
                    content = await self._call_with_retry(self._local_chat, p, resolved_model, payload, stream=False)
                elif p == "together":
                    content = await self._call_with_retry(self._together_chat, p, resolved_model, payload, stream=False)
                elif p == "openrouter":
                    content = await self._call_with_retry(self._openrouter_chat, p, resolved_model, payload, stream=False)
                else:
                    content = await self._call_with_retry(self._emergent_chat, p, resolved_model, resolved_model, norm_messages, stream=False)
                return {"provider": p, "model": resolved_model, "content": content}
            except Exception as e:
                errors.append(self._err(p, 502, "upstream_error", str(e)))
                continue
        raise HTTPException(status_code=503, detail={"message": "All providers unavailable", "chain": errors})

    async def stream(self, messages: List[Dict[str, Any]], model: str = "default", provider: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, image_url: Optional[str] = None, image_b64: Optional[str] = None, image_mime: Optional[str] = None) -> AsyncGenerator[str, None]:
        order_default = ["local", "together", "openrouter", "emergent"]
        order = []
        if provider in order_default:
            order = [provider] + [p for p in order_default if p != provider] if self.enable_fallback else [provider]
        else:
            order = order_default

        last_errors = []
        for p in order:
            if not self._breaker_allows(p):
                continue
            resolved_model = self._resolve_model(p, model)
            norm_messages = self._to_provider_messages(messages, image_url=image_url, image_b64=image_b64, image_mime=image_mime)
            payload = {"model": resolved_model, "messages": norm_messages}
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            LLM_REQUESTS.labels(provider=p, model=resolved_model).inc()
            start = time.time()
            try:
                if p == "local":
                    gen = await self._call_with_retry(self._local_chat, p, resolved_model, payload, stream=True)
                elif p == "together":
                    gen = await self._call_with_retry(self._together_chat, p, resolved_model, payload, stream=True)
                elif p == "openrouter":
                    gen = await self._call_with_retry(self._openrouter_chat, p, resolved_model, payload, stream=True)
                else:
                    gen = await self._call_with_retry(self._emergent_chat, p, resolved_model, resolved_model, norm_messages, stream=True)
                # Yield chunks
                async for chunk in gen:
                    if chunk:
                        yield chunk
                telemetry.record_latency(p, resolved_model, time.time() - start)
                return
            except Exception as e:
                telemetry.record_error(p, resolved_model, "upstream_error", 502, str(e))
                last_errors.append(self._err(p, 502, "upstream_error", str(e)))
                continue
        # if all failed, emit error as last event
        yield "\n[ERROR]: All providers unavailable"

# Async sleep helper
async def asyncio_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)

llm = LLMOrchestrator()

# Core routes
@api_router.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Hello World"}

@api_router.get("/health", response_model=HealthResponse)
async def health():
    try:
        await db.command("ping")
        mongo_status = "ok"
    except Exception as e:
        LOGGER.error(f"Mongo ping failed: {e}")
        mongo_status = "down"
    return HealthResponse(status="ok", mongo=mongo_status, time=datetime.utcnow())

@api_router.get("/version", response_model=VersionResponse)
async def version():
    return VersionResponse(version=APP_VERSION, time=datetime.utcnow())

@api_router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@api_router.get("/providers/status")
async def providers_status():
    return {
        "local": bool(llm.local_base),
        "together": bool(llm.together_key),
        "openrouter": bool(llm.openrouter_key),
        "emergent": bool(llm.emergent_client),
        "breakers": llm.breakers,
    }

@api_router.get("/providers/telemetry")
async def providers_telemetry():
    return telemetry.snapshot()

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    try:
        status_obj = StatusCheck(**input.dict())
        await db.status_checks.insert_one(status_obj.dict())
        return status_obj
    except Exception as e:
        LOGGER.exception("Failed creating status check")
        raise HTTPException(status_code=503, detail="Database error")

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None, description="Filter by client_name contains"),
):
    try:
        filter_q: Dict[str, Any] = {}
        if q:
            filter_q["client_name"] = {"$regex": q, "$options": "i"}
        cursor = db.status_checks.find(filter_q, projection={"_id": 0}).sort("timestamp", -1).skip(offset).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [StatusCheck(**strip_mongo_id(d)) for d in docs]
    except Exception as e:
        LOGGER.exception("Failed fetching status checks")
        raise HTTPException(status_code=503, detail="Database error")

@api_router.get("/status/count", response_model=StatusCountResponse)
async def status_count():
    try:
        total = await db.status_checks.count_documents({})
        distinct_clients_list = await db.status_checks.distinct("client_name")
        return StatusCountResponse(total=total, distinct_clients=len(distinct_clients_list))
    except Exception as e:
        LOGGER.exception("Failed computing counts")
        raise HTTPException(status_code=503, detail="Database error")

@api_router.get("/status/export")
async def export_status(client: Optional[str] = Query(default=None), limit: int = Query(10000, ge=1, le=100000)):
    try:
        query: Dict[str, Any] = {}
        if client:
            query["client_name"] = client
        cursor = db.status_checks.find(query, projection={"_id": 0}).sort("timestamp", -1).limit(limit)
        rows = await cursor.to_list(length=limit)
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "client_name", "timestamp"])
        for r in rows:
            writer.writerow([r.get("id"), r.get("client_name"), r.get("timestamp")])
        csv_bytes = output.getvalue()
        headers = {"Content-Disposition": f"attachment; filename=export_status.csv"}
        return Response(content=csv_bytes, media_type="text/csv", headers=headers)
    except Exception as e:
        LOGGER.exception("Failed exporting CSV")
        raise HTTPException(status_code=503, detail="Export failed")

@api_router.post("/status/purge")
async def purge_status(req: PurgeRequest):
    try:
        query: Dict[str, Any] = {}
        if req.client_name:
            query["client_name"] = req.client_name
        if req.older_than_hours:
            cutoff = datetime.utcnow() - timedelta(hours=req.older_than_hours)
            query["timestamp"] = {"$lt": cutoff}
        if not query:
            if os.environ.get("UNSAFE_PURGE_ALL", "false").lower() != "true":
                raise HTTPException(status_code=400, detail="Refusing to purge all without UNSAFE_PURGE_ALL=true")
        res = await db.status_checks.delete_many(query)
        return {"deleted": res.deleted_count}
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("Failed purging data")
        raise HTTPException(status_code=503, detail="Purge failed")

# Admin config endpoint
@api_router.get("/config")
async def get_config():
    admin_enabled = os.environ.get("ADMIN_ENABLED", "false").lower() == "true"
    return {"admin_enabled": admin_enabled}

# AI endpoints
@api_router.post("/ai/chat")
async def ai_chat(req: AIChatRequest):
    if LOG_SAMPLES:
        try:
            LOGGER.info(f"ai_chat sample: {str(req.dict())[:1024]}")
        except Exception:
            pass
    try:
        msgs = [m.dict() for m in req.messages]
        result = await llm.chat(
            messages=msgs,
            model=req.model or "default",
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            provider=req.provider,
            image_url=req.image_url,
            image_b64=req.image_b64,
            image_mime=req.image_mime,
        )
        return {"provider": result["provider"], "model": result["model"], "content": result["content"]}
    except HTTPException as he:
        raise he
    except Exception as e:
        LOGGER.exception("AI chat failed")
        raise HTTPException(status_code=503, detail={"message": "AI chat failed", "error": str(e)})

@api_router.post("/ai/chat/stream")
async def ai_chat_stream(req: AIChatRequest):
    if LOG_SAMPLES:
        try:
            LOGGER.info(f"ai_chat_stream sample: {str(req.dict())[:1024]}")
        except Exception:
            pass
    async def event_gen() -> AsyncGenerator[bytes, None]:
        try:
            msgs = [m.dict() for m in req.messages]
            async for chunk in llm.stream(
                messages=msgs,
                model=req.model or "default",
                provider=req.provider,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                image_url=req.image_url,
                image_b64=req.image_b64,
                image_mime=req.image_mime,
            ):
                if chunk:
                    yield f"data: {chunk}\n\n".encode()
            yield b"data: [DONE]\n\n"
        except Exception as e:
            err = {"message": str(e)}
            yield f"data: {err}\n\n".encode()
            yield b"data: [DONE]\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")

@api_router.post("/ai/summarize")
async def ai_summarize(req: AISummaryRequest):
    try:
        since = datetime.utcnow() - timedelta(hours=req.hours or 24)
        cursor = db.status_checks.find({"timestamp": {"$gte": since}}, projection={"_id": 0}).sort("timestamp", -1).limit(req.limit or 200)
        rows = await cursor.to_list(length=req.limit or 200)
        if not rows:
            return {"provider": None, "model": req.model, "content": "No status events in the selected window."}
        lines = [f"{r['timestamp']} - {r['client_name']} - {r['id']}" for r in rows]
        prompt = (
            "You are an SRE assistant. Given recent system status events (client name and timestamp), "
            "write a concise summary (max 200 words) with notable trends, spikes, and any anomalies. "
            "End with 3 bullet-point action items.\n\nRecent Events:\n" + "\n".join(lines)
        )
        messages = [{"role": "user", "content": prompt}]
        result = await llm.chat(messages, model=req.model or "default", temperature=req.temperature or 0.3, max_tokens=300)
        return {"provider": result["provider"], "model": result["model"], "content": result["content"]}
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("AI summarize failed")
        raise HTTPException(status_code=503, detail="AI summarize failed")

@api_router.post("/ai/insights")
async def ai_insights(req: AIInsightsRequest):
    try:
        since = datetime.utcnow() - timedelta(hours=req.hours or 24)
        rows = await db.status_checks.aggregate([
            {"$match": {"timestamp": {"$gte": since}}},
            {"$group": {"_id": "$client_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 100}
        ]).to_list(length=100)
        bullets = [f"{r['_id']}: {r['count']} events" for r in rows]
        prompt = (
            "Given the following client activity counts, identify anomalies, outliers, and potential root causes. "
            "Return bullet points only.\n\nCounts:\n" + "\n".join(bullets)
        )
        messages = [{"role": "user", "content": prompt}]
        result = await llm.chat(messages, model="default", temperature=0.2, max_tokens=250)
        return {"provider": result["provider"], "model": result["model"], "content": result["content"]}
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("AI insights failed")
        raise HTTPException(status_code=503, detail="AI insights failed")

# Mount router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    await init_indexes()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()