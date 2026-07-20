"""Periodic drift-monitoring DAG.

Runs independently of the training pipeline (movie_recommender_pipeline)
on a short interval, comparing live /recommend traffic (logged by the
FastAPI service to prediction_logs/predictions.jsonl) against the
training reference distribution. Writes drift_reports/drift_report.html
and monitoring/drift_status.json, which the API's /drift-metrics
endpoint exposes to Prometheus/Grafana.
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
    description="Compare live recommendation traffic against the training reference distribution",
    default_args=default_args,
    schedule_interval=timedelta(minutes=30),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "movie-recommender", "monitoring"],
) as dag:

    check_drift = BashOperator(
        task_id="check_drift",
        bash_command=f"cd {PROJECT_DIR} && python3 monitoring/drift_detection.py",
    )
