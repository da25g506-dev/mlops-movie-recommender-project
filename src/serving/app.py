"""FastAPI serving layer for the movie recommender.

Loads the current Production model from the MLflow Model Registry at
start-up and exposes:
  - GET /recommend/{user_id}?k=10  -> top-k recommended movies
  - GET /health                    -> readiness probe
  - GET /metrics                   -> Prometheus exposition (via instrumentator)

Every recommendation request is published as a JSON event to the Kafka
topic `recommendation-events` rather than written directly to a shared
log file - this decouples the always-on API from the batch monitoring
pipeline (the API keeps serving even if Kafka/monitoring is down, and
multiple API replicas could publish to the same topic without a shared
filesystem). `streaming/kafka_consumer.py` (run from the `drift_monitoring`
Airflow DAG) drains the topic into prediction_logs/predictions.jsonl,
which the drift-detection job (Stage 6) then reads.
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from kafka import KafkaProducer
from kafka.errors import KafkaError
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.serving.model_loader import service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIFT_STATUS_PATH = PROJECT_ROOT / "monitoring" / "drift_status.json"
FREQUENCY_STATUS_PATH = PROJECT_ROOT / "monitoring" / "recommendation_frequency.json"

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")
RECOMMENDATION_EVENTS_TOPIC = "recommendation-events"

_producer: KafkaProducer | None = None


def _get_producer() -> KafkaProducer:
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    return _producer


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    _get_producer()
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


def _publish_prediction(user_id: int, k: int, recommendations: list, latency_ms: float) -> None:
    record = {
        "timestamp": time.time(),
        "user_id": user_id,
        "k": k,
        "num_recommendations": len(recommendations),
        "movie_ids": [r["movie_id"] for r in recommendations],
        "latency_ms": latency_ms,
    }
    try:
        _get_producer().send(RECOMMENDATION_EVENTS_TOPIC, record)
    except KafkaError:
        # Logging is a monitoring side-channel, not a serving dependency -
        # a Kafka outage must not fail the recommendation request itself.
        logger.exception("Failed to publish prediction event to Kafka")


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

    _publish_prediction(user_id, k, recommendations, latency_ms)
    return {"user_id": user_id, "k": k, "recommendations": recommendations}


@app.get("/drift-metrics")
def drift_metrics():
    """Exposes the latest offline drift-detection result (produced by
    monitoring/drift_detection.py) and the latest recommendation-
    concentration result (produced by beam_jobs/aggregate_recommendations.py)
    as Prometheus gauges, so Grafana/Prometheus can alert on either without
    needing to run Evidently/Beam themselves."""
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
    top_movie_share = Gauge(
        "recommender_top_movie_share",
        "Share of all logged recommendations taken by the single most-recommended movie "
        "(computed by the Beam aggregation job) - a high value indicates the model is "
        "narrowing in on a small slice of the catalog.",
        registry=registry,
    )

    if DRIFT_STATUS_PATH.exists():
        status = json.loads(DRIFT_STATUS_PATH.read_text())
        dataset_drift.set(1 if status.get("dataset_drift") else 0)
        drift_share.set(status.get("drift_share", 0.0))
        samples.set(status.get("n_samples", 0))

    if FREQUENCY_STATUS_PATH.exists():
        frequency_status = json.loads(FREQUENCY_STATUS_PATH.read_text())
        top_movie_share.set(frequency_status.get("top_movie_share", 0.0))

    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
