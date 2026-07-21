"""Drains the `recommendation-events` Kafka topic into
prediction_logs/predictions.jsonl.

The FastAPI service (src/serving/app.py) publishes one event per
/recommend call to Kafka instead of writing the shared log file
directly (see app.py's docstring for the "why Kafka" rationale). This
script is the other half of that decoupling: a bounded batch job,
triggered periodically by the `drift_monitoring` Airflow DAG, that
drains whatever has accumulated on the topic and appends it to the same
JSONL file the drift-detection job (monitoring/drift_detection.py) and
the Beam aggregation job (beam_jobs/aggregate_recommendations.py) already
read. It exits once the topic goes quiet rather than running forever,
since it's invoked as a bounded Airflow task, not a long-lived daemon.
"""
import json
import logging
import os
from pathlib import Path
from typing import Iterable

from kafka import KafkaConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_LOG_PATH = PROJECT_ROOT / "prediction_logs" / "predictions.jsonl"

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")
RECOMMENDATION_EVENTS_TOPIC = "recommendation-events"
CONSUMER_GROUP_ID = "prediction-log-drain"
CONSUMER_TIMEOUT_MS = 5000  # stop once the topic has been quiet this long


def drain(consumer: Iterable, log_path: Path) -> int:
    """Appends the value of every message currently available on `consumer`
    to `log_path` as a JSON line. `consumer` just needs to be iterable and
    yield objects with a `.value` attribute (a dict), matching both the
    real KafkaConsumer's ConsumerRecord and simple test doubles."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(log_path, "a") as f:
        for message in consumer:
            f.write(json.dumps(message.value) + "\n")
            count += 1
    return count


def main() -> None:
    consumer = KafkaConsumer(
        RECOMMENDATION_EVENTS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=CONSUMER_TIMEOUT_MS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    count = drain(consumer, PREDICTION_LOG_PATH)
    consumer.close()
    logger.info("Drained %d recommendation events into %s", count, PREDICTION_LOG_PATH)


if __name__ == "__main__":
    main()
