import json
from fastapi import APIRouter, HTTPException
from core.redis_client import redis_client

router = APIRouter()


@router.get("/status/{job_id}")
def get_status(job_id: str):
    """Get the processing status of an uploaded document."""
    meta_raw = redis_client.hgetall(f"job:{job_id}")
    results_raw = redis_client.lrange(f"job:{job_id}:results", 0, -1)
    
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    results = [json.loads(r) for r in results_raw] if isinstance(results_raw, list) else []
    
    total = int(meta.get("total_files", 0) or meta.get(b"total_files", 0))
    completed = len(results)

    if total == 0:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    status = "completed" if completed >= total else "processing"

    return {
        "job_id": job_id,
        "status": status,
        "completed": completed,
        "total": total,
        "results": results
    }