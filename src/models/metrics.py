"""Evaluation metrics shared across all recommender models.

Two families of metrics are computed:
  - Rating-prediction accuracy: RMSE, MAE (how close predicted ratings are
    to actual ratings on the held-out test set).
  - Top-K ranking quality: Precision@K, Recall@K (whether the model's top-K
    recommended movies for a user actually appear among the movies that
    user rated highly in the held-out test set).
"""
from typing import Dict, List, Sequence

import numpy as np


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_recall_at_k(
    recommendations: Dict[int, List[int]], relevant_items: Dict[int, set], k: int = 10
) -> Dict[str, float]:
    """recommendations: user_id -> ranked list of recommended movie_ids (top-K already).
    relevant_items: user_id -> set of movie_ids the user actually liked (rating >= threshold)
    in the held-out test set.

    Only users present in both dicts are scored, since users with no
    relevant test-set items can't meaningfully contribute to precision/recall.
    """
    precisions, recalls = [], []
    for user_id, relevant in relevant_items.items():
        if not relevant:
            continue
        recs = recommendations.get(user_id, [])[:k]
        if not recs:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        hits = len(set(recs) & relevant)
        precisions.append(hits / len(recs))
        recalls.append(hits / len(relevant))

    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
    }
