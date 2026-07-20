"""Train and compare three recommender models against MLflow.

Models:
  1. Popularity baseline  - non-personalized, global movie-average rating.
  2. SVD (scikit-surprise) - classic explicit-feedback collaborative filtering.
  3. ALS (implicit)        - implicit-feedback matrix factorization.

Each model is trained on a per-user temporal train split and evaluated on
the held-out per-user test split. Every run (params, metrics, and the
serialized model artifact) is logged to MLflow under the experiment
"movie-recommender". Run `python -m src.models.evaluate` afterwards to
compare runs and register the best one.
"""
import logging
import pickle
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc

from src.models.als_model import ALSModel
from src.models.baseline_popularity import PopularityModel
from src.models.data_utils import (
    build_movie_catalog,
    build_relevant_items_map,
    build_user_items_map,
    load_ratings,
    train_test_split_by_user,
)
from src.models.metrics import mae, precision_recall_at_k, rmse
from src.models.recommender_pyfunc import RecommenderPyfunc
from src.models.svd_model import SVDModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "movie-recommender"
TOP_K = 10


def _log_pyfunc_model(model, artifact_path: str) -> None:
    """Pickle the raw model, then log it wrapped as an MLflow pyfunc model
    (proper MLmodel format) so it's servable via mlflow.pyfunc.load_model
    and registerable in the Model Registry."""
    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = Path(tmp) / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=RecommenderPyfunc(),
            artifacts={"model": str(pkl_path)},
            pip_requirements=["scikit-surprise==1.1.5", "implicit==0.7.2", "pandas==2.2.2", "numpy==1.26.4", "scipy==1.13.1"],
        )


def evaluate_rating_predictions(model, test_df) -> dict:
    y_true = test_df["rating"].to_numpy()
    y_pred = [model.predict_rating(u, m) for u, m in zip(test_df["user_id"], test_df["movie_id"])]
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}


def evaluate_ranking(model, test_users, user_items_map, relevant_items_map, k=TOP_K) -> dict:
    recs = model.recommend_batch(test_users, k, user_items_map)
    return precision_recall_at_k(recs, relevant_items_map, k=k)


def train_popularity(train_df, test_df, test_users, user_items_map, relevant_items_map):
    with mlflow.start_run(run_name="popularity_baseline"):
        params = {"min_ratings": 20}
        mlflow.log_params(params)
        mlflow.set_tag("model_family", "popularity_baseline")

        model = PopularityModel(**params).fit(train_df)

        metrics = evaluate_rating_predictions(model, test_df)
        metrics.update(evaluate_ranking(model, test_users, user_items_map, relevant_items_map))
        mlflow.log_metrics(metrics)
        _log_pyfunc_model(model, "model")

        logger.info("Popularity baseline metrics: %s", metrics)
        return metrics


def train_svd(train_df, test_df, test_users, user_items_map, relevant_items_map):
    with mlflow.start_run(run_name="svd_collaborative_filtering"):
        params = {"n_factors": 50, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}
        mlflow.log_params(params)
        mlflow.set_tag("model_family", "svd")

        model = SVDModel(**params, random_state=42).fit(train_df)

        metrics = evaluate_rating_predictions(model, test_df)
        metrics.update(evaluate_ranking(model, test_users, user_items_map, relevant_items_map))
        mlflow.log_metrics(metrics)
        _log_pyfunc_model(model, "model")

        logger.info("SVD metrics: %s", metrics)
        return metrics


def train_als(train_df, test_df, test_users, user_items_map, relevant_items_map):
    with mlflow.start_run(run_name="als_matrix_factorization"):
        params = {"factors": 50, "regularization": 0.01, "iterations": 15}
        mlflow.log_params(params)
        mlflow.set_tag("model_family", "als")

        model = ALSModel(**params, random_state=42).fit(train_df)

        # RMSE/MAE are not meaningful for implicit ALS (see ALSModel docstring);
        # only ranking quality is logged for this model family.
        metrics = evaluate_ranking(model, test_users, user_items_map, relevant_items_map)
        mlflow.log_metrics(metrics)
        _log_pyfunc_model(model, "model")

        logger.info("ALS metrics: %s", metrics)
        return metrics


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Loading processed ratings dataset")
    ratings = load_ratings()
    train_df, test_df = train_test_split_by_user(ratings, test_frac=0.2, seed=42)
    logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    user_items_map = build_user_items_map(train_df)
    relevant_items_map = build_relevant_items_map(test_df)
    test_users = list(relevant_items_map.keys())
    build_movie_catalog(ratings)  # sanity check the catalog builds without error

    results = {}
    results["popularity"] = train_popularity(train_df, test_df, test_users, user_items_map, relevant_items_map)
    results["svd"] = train_svd(train_df, test_df, test_users, user_items_map, relevant_items_map)
    results["als"] = train_als(train_df, test_df, test_users, user_items_map, relevant_items_map)

    logger.info("=== Summary ===")
    for name, metrics in results.items():
        logger.info("%s: %s", name, metrics)


if __name__ == "__main__":
    main()
