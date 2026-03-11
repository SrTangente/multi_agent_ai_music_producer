"""FastAPI server for the Music Producer API.

Provides REST API endpoints for:
- Music generation job submission
- Job status queries
- Health and readiness checks
- Prometheus metrics
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Metrics
REQUEST_COUNT = Counter(
    "music_producer_requests_total",
    "Total requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "music_producer_request_duration_seconds",
    "Request latency",
    ["endpoint"],
)
QUEUE_LENGTH = Gauge(
    "music_producer_queue_length",
    "Number of jobs in queue",
)
JOBS_COMPLETED = Counter(
    "music_producer_jobs_completed_total",
    "Total completed jobs",
)
JOB_FAILURES = Counter(
    "music_producer_job_failures_total",
    "Total failed jobs",
)
GENERATION_DURATION = Histogram(
    "music_producer_generation_duration_seconds",
    "Music generation duration",
    ["segment_type"],
    buckets=[10, 30, 60, 120, 300, 600, 1200],
)


# Request/Response models
class GenerationRequest(BaseModel):
    """Request to generate music."""
    
    prompt: str = Field(..., description="Music generation prompt")
    reference_paths: list[str] = Field(default=[], description="Reference audio file paths")
    duration: int = Field(default=120, ge=10, le=600, description="Target duration in seconds")
    callback_url: str | None = Field(default=None, description="Webhook URL for completion notification")


class GenerationResponse(BaseModel):
    """Response for generation request."""
    
    job_id: str
    status: str
    message: str
    created_at: datetime


class JobStatus(BaseModel):
    """Job status response."""
    
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    output_path: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    uptime_seconds: float


# Global state (in production, use Redis)
_jobs: dict[str, dict[str, Any]] = {}
_start_time = datetime.utcnow()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Starting Music Producer API...")
    yield
    # Shutdown
    print("Shutting down Music Producer API...")


app = FastAPI(
    title="Music Producer API",
    description="Multi-Agent AI Music Producer REST API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for liveness probe."""
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
    )


@app.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Readiness check endpoint for readiness probe."""
    # In production, check dependencies (Redis, model loaded, etc.)
    return {"status": "ready"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/api/v1/generate", response_model=GenerationResponse)
async def generate_music(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
) -> GenerationResponse:
    """Submit a music generation job."""
    with REQUEST_LATENCY.labels(endpoint="generate").time():
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "prompt": request.prompt,
            "reference_paths": request.reference_paths,
            "duration": request.duration,
            "callback_url": request.callback_url,
            "output_path": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        
        QUEUE_LENGTH.inc()
        REQUEST_COUNT.labels(endpoint="generate", status="accepted").inc()
        
        # In production, submit to Redis queue for K8s Job pickup
        background_tasks.add_task(_process_job, job_id)
        
        return GenerationResponse(
            job_id=job_id,
            status="pending",
            message="Job submitted successfully",
            created_at=now,
        )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Get status of a generation job."""
    if job_id not in _jobs:
        REQUEST_COUNT.labels(endpoint="job_status", status="not_found").inc()
        raise HTTPException(status_code=404, detail="Job not found")
    
    REQUEST_COUNT.labels(endpoint="job_status", status="success").inc()
    job = _jobs[job_id]
    return JobStatus(**job)


@app.get("/api/v1/jobs")
async def list_jobs(
    status: str | None = None,
    limit: int = 50,
) -> list[JobStatus]:
    """List generation jobs."""
    jobs = list(_jobs.values())
    
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return [JobStatus(**j) for j in jobs[:limit]]


async def _process_job(job_id: str) -> None:
    """Process a music generation job.
    
    In production, this would submit a K8s Job instead of processing inline.
    """
    job = _jobs.get(job_id)
    if not job:
        return
    
    try:
        job["status"] = "running"
        job["updated_at"] = datetime.utcnow()
        QUEUE_LENGTH.dec()
        
        # Simulate processing (in production, K8s Job handles this)
        await asyncio.sleep(5)
        
        # Mark complete
        job["status"] = "completed"
        job["progress"] = 1.0
        job["output_path"] = f"/app/output/{job_id}/final.wav"
        job["updated_at"] = datetime.utcnow()
        
        JOBS_COMPLETED.inc()
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = datetime.utcnow()
        JOB_FAILURES.inc()


def main() -> None:
    """Run the API server."""
    import uvicorn
    
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        workers=int(os.environ.get("WORKERS", 1)),
    )


if __name__ == "__main__":
    main()
