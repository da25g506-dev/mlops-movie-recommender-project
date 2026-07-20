"""Drift detection for the movie recommender's live traffic.

Reference distribution: per-movie stats (avg rating, rating count) for
the whole training catalog - i.e. what "the average recommended movie"
should look like if the model samples broadly across the catalog.

Current distribution: the same per-movie stats, but only for movies
that were actually recommended in production (from the prediction log
written by src/serving/app.py). If the model starts drifting toward a
narrow slice of the catalog (e.g. always recommending the same handful
of blockbusters, or degrading into recommending obscure/low-count
titles), this shows up as distribution drift on movie_rating_count
and/or movie_avg_rating.

Writes an HTML report to drift_reports/ and a small JSON summary to
monitoring/drift_status.json that the API's /drift-metrics endpoint
exposes to Prometheus.
"""
import json
import logging
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PARQUET = PROJECT_ROOT / "data" / "processed" / "ratings_features.parquet"
PREDICTION_LOG_PATH = PROJECT_ROOT / "prediction_logs" / "predictions.jsonl"
DRIFT_REPORTS_DIR = PROJECT_ROOT / "drift_reports"
DRIFT_STATUS_PATH = PROJECT_ROOT / "monitoring" / "drift_status.json"

FEATURE_COLUMNS = ["movie_rating_count", "movie_avg_rating"]
MIN_CURRENT_SAMPLES = 30


def load_movie_stats() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_PARQUET, columns=["movie_id"] + FEATURE_COLUMNS)
    return df.drop_duplicates(subset="movie_id").set_index("movie_id")


def load_recommended_movie_ids() -> list:
    if not PREDICTION_LOG_PATH.exists():
        return []
    movie_ids = []
    with open(PREDICTION_LOG_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            movie_ids.extend(record.get("movie_ids", []))
    return movie_ids


def build_current_df(movie_stats: pd.DataFrame, movie_ids: list) -> pd.DataFrame:
    valid_ids = [mid for mid in movie_ids if mid in movie_stats.index]
    return movie_stats.loc[valid_ids, FEATURE_COLUMNS].reset_index(drop=True)


def extract_drift_summary(report: Report) -> dict:
    result = report.as_dict()
    drift_metric = next(
        m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric"
    )
    return {
        "dataset_drift": bool(drift_metric["result"]["dataset_drift"]),
        "drift_share": float(drift_metric["result"]["drift_share"]),
        "number_of_drifted_columns": int(drift_metric["result"]["number_of_drifted_columns"]),
    }


def main() -> None:
    movie_stats = load_movie_stats()
    reference_df = movie_stats[FEATURE_COLUMNS].reset_index(drop=True)

    recommended_ids = load_recommended_movie_ids()
    current_df = build_current_df(movie_stats, recommended_ids)

    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if len(current_df) < MIN_CURRENT_SAMPLES:
        logger.info(
            "Only %d recommended-movie samples logged (need >= %d) - skipping drift check.",
            len(current_df), MIN_CURRENT_SAMPLES,
        )
        DRIFT_STATUS_PATH.write_text(json.dumps({
            "dataset_drift": False,
            "drift_share": 0.0,
            "number_of_drifted_columns": 0,
            "n_samples": len(current_df),
            "status": "insufficient_data",
        }))
        return

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    report_path = DRIFT_REPORTS_DIR / "drift_report.html"
    report.save_html(str(report_path))
    logger.info("Saved drift report to %s", report_path)

    summary = extract_drift_summary(report)
    summary["n_samples"] = len(current_df)
    summary["status"] = "ok"
    DRIFT_STATUS_PATH.write_text(json.dumps(summary))
    logger.info("Drift summary: %s", summary)


if __name__ == "__main__":
    main()
