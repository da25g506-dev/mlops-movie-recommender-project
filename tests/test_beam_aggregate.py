"""Tests for beam_jobs/aggregate_recommendations.py.

Runs the real Beam DirectRunner pipeline (in-process, no cluster) over
a small fixture file, so these tests exercise the actual pipeline
shape rather than a mocked-out stand-in.
"""
import json

from beam_jobs.aggregate_recommendations import run


def _write_log(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_run_computes_counts_and_top_movie_share(tmp_path):
    input_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "recommendation_frequency.json"
    _write_log(input_path, [
        {"movie_ids": [101, 102]},
        {"movie_ids": [101]},
        {"movie_ids": [103, 101]},
    ])

    result = run(input_path, output_path)

    assert result["counts"] == {"101": 3, "102": 1, "103": 1}
    assert result["total"] == 5
    assert result["top_movie_share"] == 3 / 5
    assert json.loads(output_path.read_text()) == result


def test_run_handles_missing_input_file(tmp_path):
    input_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "recommendation_frequency.json"

    result = run(input_path, output_path)

    assert result == {"counts": {}, "total": 0, "top_movie_share": 0.0}
    assert json.loads(output_path.read_text()) == result


def test_run_handles_empty_input_file(tmp_path):
    input_path = tmp_path / "predictions.jsonl"
    input_path.write_text("")
    output_path = tmp_path / "recommendation_frequency.json"

    result = run(input_path, output_path)

    assert result == {"counts": {}, "total": 0, "top_movie_share": 0.0}


def test_run_handles_uniform_distribution(tmp_path):
    input_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "recommendation_frequency.json"
    _write_log(input_path, [
        {"movie_ids": [101]},
        {"movie_ids": [102]},
    ])

    result = run(input_path, output_path)

    assert result["counts"] == {"101": 1, "102": 1}
    assert result["top_movie_share"] == 0.5
