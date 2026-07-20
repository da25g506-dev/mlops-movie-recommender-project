"""Tests for monitoring/drift_detection.py.

Avoids exercising the real Evidently Report.run() (slow, and not needed
to verify our own log-parsing/threshold logic) - only the insufficient-data
path in main() is tested end-to-end.
"""
import json

import pandas as pd

from monitoring import drift_detection


def test_load_recommended_movie_ids_reads_jsonl(tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text(
        json.dumps({"movie_ids": [1, 2, 3]}) + "\n"
        + json.dumps({"movie_ids": [4, 5]}) + "\n"
    )
    monkeypatch.setattr(drift_detection, "PREDICTION_LOG_PATH", log_path)

    assert drift_detection.load_recommended_movie_ids() == [1, 2, 3, 4, 5]


def test_load_recommended_movie_ids_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_detection, "PREDICTION_LOG_PATH", tmp_path / "missing.jsonl")

    assert drift_detection.load_recommended_movie_ids() == []


def test_build_current_df_drops_unknown_movie_ids():
    movie_stats = pd.DataFrame(
        {"movie_rating_count": [10, 20], "movie_avg_rating": [3.5, 4.0]},
        index=pd.Index([1, 2], name="movie_id"),
    )

    current_df = drift_detection.build_current_df(movie_stats, [1, 2, 999])

    assert len(current_df) == 2
    assert list(current_df.columns) == drift_detection.FEATURE_COLUMNS


def test_main_writes_insufficient_data_status_below_threshold(tmp_path, monkeypatch):
    movie_stats = pd.DataFrame(
        {
            "movie_id": [1, 2],
            "movie_rating_count": [10, 20],
            "movie_avg_rating": [3.5, 4.0],
        }
    )
    parquet_path = tmp_path / "ratings_features.parquet"
    movie_stats.to_parquet(parquet_path)

    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text(json.dumps({"movie_ids": [1, 2]}) + "\n")

    status_path = tmp_path / "drift_status.json"
    reports_dir = tmp_path / "drift_reports"

    monkeypatch.setattr(drift_detection, "PROCESSED_PARQUET", parquet_path)
    monkeypatch.setattr(drift_detection, "PREDICTION_LOG_PATH", log_path)
    monkeypatch.setattr(drift_detection, "DRIFT_STATUS_PATH", status_path)
    monkeypatch.setattr(drift_detection, "DRIFT_REPORTS_DIR", reports_dir)

    drift_detection.main()

    status = json.loads(status_path.read_text())
    assert status["status"] == "insufficient_data"
    assert status["n_samples"] == 2
    assert status["dataset_drift"] is False
