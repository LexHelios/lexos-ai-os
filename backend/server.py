from fastapi import FastAPI, APIRouter, HTTPException, Query, Request, Response
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import httpx
import time
import csv
from io import StringIO
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# App init
app = FastAPI()
api_router = APIRouter(prefix="/api")

APP_VERSION = "1.1.0"
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
    # simple path normalization: collapse ids
    norm_path = path.replace(request.app.router.prefix if hasattr(request.app, 'router') else "", "")
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
        LOGGER.info(f"trace_id=%s method=%s path=%s status=%s latency_ms=%.2f", trace_id, method, path, (response.status_code if 'response' in locals() else 'NA'), duration * 1000)

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
        self.local_base = os.environ.get("LLM_LOCAL_BASE_URL")  # e.g. http://127.0.0.1:8080
        self.together_base = os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
        self.together_key = os.environ.get("TOGETHER_API_KEY")
        self.openrouter_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "45"))

    async def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()

    async def chat(self, messages: List[Dict[str, str]], model: str = "default", temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Provider order: Local -> Together -> OpenRouter
        # Local
        if self.local_base:
            try:
                data = await self._post_json(f"{self.local_base}/v1/chat/completions", payload)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return {"provider": "local", "model": data.get("model", model), "content": content, "raw": data}
            except Exception as e:
                LOGGER.warning(f"Local LLM failed: {e}")
        # Together
        if self.together_key:
            try:
                headers = {"Authorization": f"Bearer {self.together_key}", "Content-Type": "application/json"}
                data = await self._post_json(f"{self.together_base}/chat/completions", payload, headers)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return {"provider": "together", "model": data.get("model", model), "content": content, "raw": data}
            except Exception as e:
                LOGGER.warning(f"Together.ai failed: {e}")
        # OpenRouter
        if self.openrouter_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "http://localhost"),
                    "X-Title": os.environ.get("OPENROUTER_TITLE", "Unrestricted App")
                }
                data = await self._post_json(f"{self.openrouter_base}/chat/completions", payload, headers)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return {"provider": "openrouter", "model": data.get("model", model), "content": content, "raw": data}
            except Exception as e:
                LOGGER.warning(f"OpenRouter failed: {e}")
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
        # Build CSV
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
            # if no filters, require explicit confirmation via env UNSAFE_PURGE_ALL=true
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
class AISummaryRequest(BaseModel):
    hours: Optional[int] = Field(default=24, ge=1, le=168)
    limit: Optional[int] = Field(default=200, ge=1, le=2000)
    model: Optional[str] = Field(default="default")
    temperature: Optional[float] = Field(default=0.3)

@api_router.post("/ai/summarize")
async def ai_summarize(req: AISummaryRequest):
    try:
        since = datetime.utcnow() - timedelta(hours=req.hours or 24)
        cursor = db.status_checks.find({"timestamp": {"$gte": since}}, projection={"_id": 0}).sort("timestamp", -1).limit(req.limit or 200)
        rows = await cursor.to_list(length=req.limit or 200)
        if not rows:
            return {"provider": None, "model": req.model, "content": "No status events in the selected window."}
        # Build prompt
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