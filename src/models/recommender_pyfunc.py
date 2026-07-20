"""Generic MLflow pyfunc wrapper around any of the three recommender models.

All three model classes (PopularityModel, SVDModel, ALSModel) expose the
same `recommend(user_id, k, exclude_items)` interface, so a single pyfunc
wrapper can serve any of them once they're unpickled. This lets us log
each trained model in proper MLflow Model format (not a bare pickle),
which is what `mlflow.register_model` and the Model Registry require, and
what the FastAPI serving layer loads at inference time via
`mlflow.pyfunc.load_model(...)`.

Expected model_input: a pandas DataFrame with columns:
  - user_id (int, required)
  - k (int, optional, defaults to 10)
Output: for each input row, a list of recommended movie_ids.
"""
import pickle

import mlflow.pyfunc
import pandas as pd


class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        recommendations = []
        for _, row in model_input.iterrows():
            user_id = int(row["user_id"])
            k = int(row["k"]) if "k" in row and not pd.isna(row["k"]) else 10
            recs = self.model.recommend(user_id, k, exclude_items=set())
            recommendations.append(recs)
        return recommendations
