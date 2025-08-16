from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from starlette.responses import JSONResponse
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# App init
app = FastAPI()
api_router = APIRouter(prefix="/api")

APP_VERSION = "1.0.0"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

# Routes
@api_router.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Hello World"}

@api_router.get("/health", response_model=HealthResponse)
async def health():
    try:
        # Ping DB
        await db.command("ping")
        mongo_status = "ok"
    except Exception as e:
        LOGGER.error(f"Mongo ping failed: {e}")
        mongo_status = "down"
    return HealthResponse(status="ok", mongo=mongo_status, time=datetime.utcnow())

@api_router.get("/version", response_model=VersionResponse)
async def version():
    return VersionResponse(version=APP_VERSION, time=datetime.utcnow())

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
            # simple case-insensitive contains
            filter_q["client_name"] = {"$regex": q, "$options": "i"}
        cursor = db.status_checks.find(filter_q, projection={"_id": 0}).sort("timestamp", -1).skip(offset).limit(limit)
        docs = await cursor.to_list(length=limit)
        # projection already removed _id, but keep safe
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