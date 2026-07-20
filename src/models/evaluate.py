"""Compare MLflow runs from the movie-recommender experiment and register
the best model in the MLflow Model Registry.

Selection rule: rank all runs by recall_at_10 (the ranking-quality metric
every model family logs, unlike RMSE/MAE which ALS doesn't produce), then
register the winning run's logged pyfunc model (artifact path "model") as
a new version of the "movie-recommender-prod" registered model and
promote it to stage "Production".
"""
import logging
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.artifacts
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "movie-recommender"
REGISTERED_MODEL_NAME = "movie-recommender-prod"
SELECTION_METRIC = "recall_at_10"
ARTIFACT_PATH = "model"
MODEL_FAMILIES = ["popularity_baseline", "svd", "als"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "production_model"


def export_production_model(model_uri: str) -> None:
    """Download the Production-stage model's artifacts out of MLflow's own
    artifact store into models/production_model/, so the currently-served
    model also exists as a plain, DVC-trackable local artifact (not just
    inside the MLflow server's volume)."""
    if LOCAL_MODEL_DIR.exists():
        shutil.rmtree(LOCAL_MODEL_DIR)
    LOCAL_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(LOCAL_MODEL_DIR))
    logger.info("Exported production model artifacts to %s", LOCAL_MODEL_DIR)


def get_latest_runs(client: MlflowClient, experiment_id: str):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    seen_families = set()
    latest = []
    for run in runs:
        family = run.data.tags.get("model_family")
        if family and family not in seen_families:
            seen_families.add(family)
            latest.append(run)
        if len(seen_families) == len(MODEL_FAMILIES):
            break
    return latest


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run src/models/train.py first.")

    latest_runs = get_latest_runs(client, experiment.experiment_id)
    if not latest_runs:
        raise RuntimeError("No runs found to compare.")

    logger.info("Comparing %d most recent runs (one per model family):", len(latest_runs))
    for run in latest_runs:
        family = run.data.tags.get("model_family")
        metric_val = run.data.metrics.get(SELECTION_METRIC)
        logger.info("  %-20s run_id=%s %s=%s all_metrics=%s", family, run.info.run_id, SELECTION_METRIC, metric_val, run.data.metrics)

    scored = [r for r in latest_runs if SELECTION_METRIC in r.data.metrics]
    best_run = max(scored, key=lambda r: r.data.metrics[SELECTION_METRIC])
    best_family = best_run.data.tags.get("model_family")

    logger.info(
        "Best model: family=%s run_id=%s %s=%.4f",
        best_family, best_run.info.run_id, SELECTION_METRIC, best_run.data.metrics[SELECTION_METRIC],
    )

    model_uri = f"runs:/{best_run.info.run_id}/{ARTIFACT_PATH}"
    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    logger.info("Registered model '%s' version %s", REGISTERED_MODEL_NAME, result.version)

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info("Promoted version %s to stage Production", result.version)

    export_production_model(model_uri)


if __name__ == "__main__":
    main()
