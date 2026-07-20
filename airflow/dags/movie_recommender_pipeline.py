"""Airflow DAG orchestrating the movie recommender pipeline end-to-end.

download_data -> spark_preprocess -> dvc_commit_processed -> train_models -> evaluate_and_register

Each task shells out to the corresponding project script inside the
Airflow worker container, which has the project repository mounted at
PROJECT_DIR and the same pinned requirements installed. Retries are
enabled since network-dependent steps (dataset download, MLflow server
availability) can transiently fail.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

default_args = {
    "owner": "movie-recommender-mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="movie_recommender_pipeline",
    description="Download -> Spark preprocess -> DVC version -> train (3 models) -> evaluate/register",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "movie-recommender"],
) as dag:

    download_data = BashOperator(
        task_id="download_data",
        bash_command=f"cd {PROJECT_DIR} && python3 src/data/download.py",
    )

    spark_preprocess = BashOperator(
        task_id="spark_preprocess",
        bash_command=f"cd {PROJECT_DIR} && python3 spark_jobs/preprocess.py",
    )

    dvc_commit_processed = BashOperator(
        task_id="dvc_commit_processed",
        bash_command=(
            f"cd {PROJECT_DIR} && python3 scripts/dvc_commit.py "
            "data/raw/ratings.dat data/raw/movies.dat data/raw/users.dat "
            "data/processed/ratings_features.parquet data/processed/ratings_features.csv"
        ),
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command=f"cd {PROJECT_DIR} && python3 -m src.models.train",
        env={"MLFLOW_TRACKING_URI": "http://mlflow:5000"},
        append_env=True,
    )

    evaluate_and_register = BashOperator(
        task_id="evaluate_and_register",
        bash_command=f"cd {PROJECT_DIR} && python3 -m src.models.evaluate",
        env={"MLFLOW_TRACKING_URI": "http://mlflow:5000"},
        append_env=True,
    )

    download_data >> spark_preprocess >> dvc_commit_processed >> train_models >> evaluate_and_register
