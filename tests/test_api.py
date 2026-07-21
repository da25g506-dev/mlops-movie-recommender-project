"""Tests for the FastAPI serving layer (src/serving/app.py).

The real app loads a model from the MLflow Model Registry at start-up,
which isn't available in a unit-test environment. We monkeypatch
`RecommenderService.load` to install a tiny fake model instead, and
monkeypatch the Kafka producer getter to a fake producer, so the tests
exercise the actual HTTP routing/validation/publishing logic without
needing a live MLflow server or Kafka broker.
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving import model_loader


class FakeModel:
    def predict(self, model_input: pd.DataFrame):
        k = int(model_input.iloc[0]["k"])
        return [list(range(101, 101 + k))]


class FakeProducer:
    def __init__(self):
        self.sent = []

    def send(self, topic, value):
        self.sent.append((topic, value))


@pytest.fixture
def client(monkeypatch, tmp_path):
    def fake_load(self):
        self._model = FakeModel()
        self._catalog = {101: "Movie A", 102: "Movie B", 103: "Movie C"}

    monkeypatch.setattr(model_loader.RecommenderService, "load", fake_load)

    from src.serving import app as app_module

    fake_producer = FakeProducer()
    monkeypatch.setattr(app_module, "_get_producer", lambda: fake_producer)

    with TestClient(app_module.app) as test_client:
        test_client.fake_producer = fake_producer
        yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_returns_titles(client):
    response = client.get("/recommend/42?k=3")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 42
    assert body["k"] == 3
    assert body["recommendations"] == [
        {"movie_id": 101, "title": "Movie A"},
        {"movie_id": 102, "title": "Movie B"},
        {"movie_id": 103, "title": "Movie C"},
    ]


def test_recommend_default_k(client):
    response = client.get("/recommend/42")
    assert response.status_code == 200
    assert response.json()["k"] == 10


@pytest.mark.parametrize("k", [0, 101, -5])
def test_recommend_rejects_invalid_k(client, k):
    response = client.get(f"/recommend/1?k={k}")
    assert response.status_code == 400


def test_recommend_publishes_prediction_event(client):
    client.get("/recommend/7?k=2")

    assert len(client.fake_producer.sent) == 1
    topic, record = client.fake_producer.sent[0]
    assert topic == "recommendation-events"
    assert record["user_id"] == 7
    assert record["k"] == 2
    assert record["movie_ids"] == [101, 102]


def test_drift_metrics_defaults_to_zero_without_status_file(client, tmp_path):
    from src.serving import app as app_module

    app_module.DRIFT_STATUS_PATH = tmp_path / "drift_status.json"
    app_module.FREQUENCY_STATUS_PATH = tmp_path / "recommendation_frequency.json"
    response = client.get("/drift-metrics")
    assert response.status_code == 200
    assert "recommender_dataset_drift 0.0" in response.text
    assert "recommender_drift_share 0.0" in response.text
    assert "recommender_drift_samples 0.0" in response.text
    assert "recommender_top_movie_share 0.0" in response.text


def test_drift_metrics_reflects_status_file(client, tmp_path):
    from src.serving import app as app_module

    status_path = tmp_path / "drift_status.json"
    status_path.write_text(json.dumps({"dataset_drift": True, "drift_share": 0.5, "n_samples": 480}))
    app_module.DRIFT_STATUS_PATH = status_path

    frequency_path = tmp_path / "recommendation_frequency.json"
    frequency_path.write_text(json.dumps({"top_movie_share": 0.3}))
    app_module.FREQUENCY_STATUS_PATH = frequency_path

    response = client.get("/drift-metrics")
    assert response.status_code == 200
    assert "recommender_dataset_drift 1.0" in response.text
    assert "recommender_drift_share 0.5" in response.text
    assert "recommender_drift_samples 480.0" in response.text
    assert "recommender_top_movie_share 0.3" in response.text
