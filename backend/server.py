from fastapi import FastAPI, APIRouter, HTTPException, Query, Request, Response
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
import uuid
from datetime import datetime, timedelta
import httpx
import time
import csv
from io import StringIO
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

APP_VERSION = "1.3.0"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Prometheus metrics
REQ_COUNTER = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "path"]) 

@app.middleware("http")
async def metrics_logging_middleware(request: Request, call_next):
    start = time.time()
    trace_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    path = request.url.path
    method = request.method
    try:
        response: Response = await call_next(request)
        status = response.status_code
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

# LLM Orchestrator
class LLMOrchestrator:
    def __init__(self):
        self.local_base = os.environ.get("LLM_LOCAL_BASE_URL")  # e.g. http://127.0.0.1:11434 or vLLM base
        self.together_base = os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
        self.together_key = os.environ.get("TOGETHER_API_KEY")
        self.openrouter_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "45"))
        self.enable_fallback = os.environ.get("ENABLE_LLM_FALLBACK", "true").lower() == "true"
        # Emergent
        self.emergent_key = os.environ.get("EMERGENT_LLM_KEY")
        self.emergent_client = EmergentLLM(api_key=self.emergent_key) if (EmergentLLM and self.emergent_key) else None

    async def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()

    async def _local_chat(self, payload_openai: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.local_base:
            return None
        # Try OpenAI-compatible first
        try:
            data = await self._post_json(f"{self.local_base}/v1/chat/completions", payload_openai)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return {"provider": "local", "model": data.get("model", payload_openai.get("model", "default")), "content": content, "raw": data}
        except Exception as e:
            # Try Ollama-compatible /api/chat
            try:
                ollama_payload = {
                    "model": payload_openai.get("model", "llama3"),
                    "messages": payload_openai.get("messages", []),
                    "stream": False
                }
                data = await self._post_json(f"{self.local_base}/api/chat", ollama_payload)
                content = data.get("message", {}).get("content")
                if content:
                    return {"provider": "local", "model": ollama_payload["model"], "content": content, "raw": data}
            except Exception as e2:
                LOGGER.warning(f"Local LLM failed (OpenAI and Ollama paths): {e} | {e2}")
        return None

    async def _together_chat(self, payload_openai: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.together_key:
            return None
        try:
            headers = {"Authorization": f"Bearer {self.together_key}", "Content-Type": "application/json"}
            data = await self._post_json(f"{self.together_base}/chat/completions", payload_openai, headers)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return {"provider": "together", "model": data.get("model", payload_openai.get("model", "default")), "content": content, "raw": data}
        except Exception as e:
            LOGGER.warning(f"Together.ai failed: {e}")
        return None

    async def _openrouter_chat(self, payload_openai: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.openrouter_key:
            return None
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "http://localhost"),
                "X-Title": os.environ.get("OPENROUTER_TITLE", "Neo System")
            }
            data = await self._post_json(f"{self.openrouter_base}/chat/completions", payload_openai, headers)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return {"provider": "openrouter", "model": data.get("model", payload_openai.get("model", "default")), "content": content, "raw": data}
        except Exception as e:
            LOGGER.warning(f"OpenRouter failed: {e}")
        return None

    async def _emergent_chat(self, payload_openai: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.emergent_client:
            return None
        try:
            # Emergent Integrations accepts OpenAI-style messages and model routing using Universal Key
            resp = self.emergent_client.chat.completions.create(
                model=payload_openai.get("model", "gpt-4"),
                messages=payload_openai.get("messages", []),
            )
            # resp.choices[0].message.content
            content = getattr(resp.choices[0].message, "content", None) if hasattr(resp, "choices") else None
            if content:
                return {"provider": "emergent", "model": payload_openai.get("model", "gpt-4"), "content": content, "raw": None}
        except Exception as e:
            LOGGER.warning(f"Emergent provider failed: {e}")
        return None

    async def chat(self, messages: List[Dict[str, Any]], model: str = "default", temperature: Optional[float] = None, max_tokens: Optional[int] = None, provider: Optional[str] = None, image_url: Optional[str] = None) -> Dict[str, Any]:
        # Build OpenAI-style payload
        formatted_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            formatted_messages.append({"role": role, "content": content})
        if image_url:
            if formatted_messages and formatted_messages[-1]["role"] == "user":
                text_part = formatted_messages[-1]["content"]
                formatted_messages[-1]["content"] = [
                    {"type": "text", "text": text_part},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            else:
                formatted_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                })

        payload = {
            "model": model,
            "messages": formatted_messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Provider routing, with Emergent at the end
        default_order = ["local", "together", "openrouter", "emergent"]
        order = []
        if provider in ("local", "together", "openrouter", "emergent"):
            order = [provider]
            if self.enable_fallback:
                for p in default_order:
                    if p not in order:
                        order.append(p)
        else:
            order = default_order

        for p in order:
            try:
                if p == "local":
                    res = await self._local_chat(payload)
                elif p == "together":
                    res = await self._together_chat(payload)
                elif p == "openrouter":
                    res = await self._openrouter_chat(payload)
                else:
                    res = await self._emergent_chat(payload)
                if res and res.get("content"):
                    return res
            except Exception as e:
                LOGGER.warning(f"Provider {p} error: {e}")
                continue

        raise HTTPException(status_code=503, detail="No LLM providers available or all failed")

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
    status = {"local": False, "together": False, "openrouter": False, "emergent": False}
    status["local"] = bool(llm.local_base)
    status["together"] = bool(llm.together_key)
    status["openrouter"] = bool(llm.openrouter_key)
    status["emergent"] = bool(llm.emergent_client)
    return status

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

class AISummaryRequest(BaseModel):
    hours: Optional[int] = Field(default=24, ge=1, le=168)
    limit: Optional[int] = Field(default=200, ge=1, le=2000)
    model: Optional[str] = Field(default="default")
    temperature: Optional[float] = Field(default=0.3)

@api_router.post("/ai/chat")
async def ai_chat(req: AIChatRequest):
    try:
        msgs = [m.dict() for m in req.messages]
        result = await llm.chat(
            messages=msgs,
            model=req.model or "default",
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            provider=req.provider,
            image_url=req.image_url,
        )
        return {"provider": result["provider"], "model": result["model"], "content": result["content"]}
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("AI chat failed")
        raise HTTPException(status_code=503, detail="AI chat failed")

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

class AIInsightsRequest(BaseModel):
    hours: Optional[int] = Field(default=24, ge=1, le=720)
    limit: Optional[int] = Field(default=500, ge=1, le=5000)

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