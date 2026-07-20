"""FastAPI serving layer for the movie recommender.

Loads the current Production model from the MLflow Model Registry at
start-up and exposes:
  - GET /recommend/{user_id}?k=10  -> top-k recommended movies
  - GET /health                    -> readiness probe
  - GET /metrics                   -> Prometheus exposition (via instrumentator)

Every recommendation request is appended as a JSON line to
prediction_logs/predictions.jsonl (mounted volume) so the drift-detection
job (Stage 6) can compare live request/response distributions against
the training reference data.
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.serving.model_loader import service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_LOG_PATH = PROJECT_ROOT / "prediction_logs" / "predictions.jsonl"
PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
DRIFT_STATUS_PATH = PROJECT_ROOT / "monitoring" / "drift_status.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    yield


app = FastAPI(title="Movie Recommender API", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class Recommendation(BaseModel):
    movie_id: int
    title: str


class RecommendResponse(BaseModel):
    user_id: int
    k: int
    recommendations: list[Recommendation]


def _log_prediction(user_id: int, k: int, recommendations: list, latency_ms: float) -> None:
    record = {
        "timestamp": time.time(),
        "user_id": user_id,
        "k": k,
        "num_recommendations": len(recommendations),
        "movie_ids": [r["movie_id"] for r in recommendations],
        "latency_ms": latency_ms,
    }
    with open(PREDICTION_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, k: int = 10):
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if k < 1 or k > 100:
        raise HTTPException(status_code=400, detail="k must be between 1 and 100")

    start = time.perf_counter()
    try:
        recommendations = service.recommend(user_id, k)
    except Exception:
        logger.exception("Recommendation failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")
    latency_ms = (time.perf_counter() - start) * 1000

    _log_prediction(user_id, k, recommendations, latency_ms)
    return {"user_id": user_id, "k": k, "recommendations": recommendations}


@app.get("/drift-metrics")
def drift_metrics():
    """Exposes the latest offline drift-detection result (produced by
    monitoring/drift_detection.py) as Prometheus gauges, so Grafana/Prometheus
    can alert on drift without needing to run Evidently themselves."""
    registry = CollectorRegistry()
    dataset_drift = Gauge(
        "recommender_dataset_drift", "1 if the latest drift check flagged dataset drift, else 0",
        registry=registry,
    )
    drift_share = Gauge(
        "recommender_drift_share", "Share of drifted columns in the latest drift check", registry=registry
    )
    samples = Gauge(
        "recommender_drift_samples", "Number of recommended-movie samples used in the latest drift check",
        registry=registry,
    )

    if DRIFT_STATUS_PATH.exists():
        status = json.loads(DRIFT_STATUS_PATH.read_text())
        dataset_drift.set(1 if status.get("dataset_drift") else 0)
        drift_share.set(status.get("drift_share", 0.0))
        samples.set(status.get("n_samples", 0))

    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
