"""Loads the current Production model from the MLflow Model Registry and
the movie catalog (id -> title) used to render human-readable
recommendations. Both are cached at process start-up.
"""
import logging
import os
from pathlib import Path
from typing import Dict

import mlflow.pyfunc
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOVIES_DAT = PROJECT_ROOT / "data" / "raw" / "movies.dat"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "movie-recommender-prod")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")


def load_movie_catalog() -> Dict[int, str]:
    rows = []
    with open(MOVIES_DAT, "r", encoding="ISO-8859-1") as f:
        for line in f:
            movie_id, title, _genres = line.rstrip("\n").split("::")
            rows.append((int(movie_id), title))
    return dict(rows)


def load_production_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


class RecommenderService:
    """Lazily loaded singleton holding the model + catalog for the API."""

    def __init__(self):
        self._model = None
        self._catalog: Dict[int, str] = {}

    def load(self) -> None:
        self._model = load_production_model()
        self._catalog = load_movie_catalog()
        logger.info("Loaded model and catalog with %d movies", len(self._catalog))

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def title_for(self, movie_id: int) -> str:
        return self._catalog.get(movie_id, "Unknown title")

    def recommend(self, user_id: int, k: int) -> list:
        model_input = pd.DataFrame({"user_id": [user_id], "k": [k]})
        predictions = self._model.predict(model_input)
        movie_ids = predictions[0]
        return [{"movie_id": int(mid), "title": self.title_for(int(mid))} for mid in movie_ids]


service = RecommenderService()
