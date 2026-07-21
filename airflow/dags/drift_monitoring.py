"""Periodic drift-monitoring DAG.

Runs independently of the training pipeline (movie_recommender_pipeline)
on a short interval:
  1. consume_kafka_events drains the `recommendation-events` Kafka topic
     (published to by the FastAPI service on every /recommend call) into
     prediction_logs/predictions.jsonl.
  2. aggregate_recommendations runs a Beam DirectRunner batch job over
     that same log, computing per-movie recommendation frequency and a
     "top_movie_share" concentration signal, written to
     monitoring/recommendation_frequency.json.
  3. check_drift compares that live traffic against the training
     reference distribution. Writes drift_reports/drift_report.html and
     monitoring/drift_status.json. Both status files are exposed by the
     API's /drift-metrics endpoint to Prometheus/Grafana.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

default_args = {
    "owner": "movie-recommender-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="drift_monitoring",
    description="Drain Kafka events -> aggregate recommendation frequency (Beam) -> compare live traffic vs. training reference",
    default_args=default_args,
    schedule_interval=timedelta(minutes=30),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "movie-recommender", "monitoring"],
) as dag:

    consume_kafka_events = BashOperator(
        task_id="consume_kafka_events",
        bash_command=f"cd {PROJECT_DIR} && python3 streaming/kafka_consumer.py",
    )

    aggregate_recommendations = BashOperator(
        task_id="aggregate_recommendations",
        bash_command=f"cd {PROJECT_DIR} && python3 beam_jobs/aggregate_recommendations.py",
    )

    check_drift = BashOperator(
        task_id="check_drift",
        bash_command=f"cd {PROJECT_DIR} && python3 monitoring/drift_detection.py",
    )

    consume_kafka_events >> aggregate_recommendations >> check_drift
