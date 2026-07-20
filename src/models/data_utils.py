"""Shared data-loading and train/test split utilities for model training.

Uses a per-user temporal split: each user's most recent `test_frac` of
ratings become the held-out test set, the rest are training data. This
avoids leaking future interactions into training and lets us evaluate
both rating-prediction (RMSE/MAE) and top-K ranking (Precision@K/Recall@K)
metrics per user.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PARQUET = PROJECT_ROOT / "data" / "processed" / "ratings_features.parquet"

RELEVANCE_THRESHOLD = 4  # rating >= this is considered a "relevant"/liked item for Precision@K/Recall@K


def load_ratings() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_PARQUET, columns=["user_id", "movie_id", "rating", "timestamp", "title"])
    return df


def train_test_split_by_user(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_id", "timestamp"])
    train_parts, test_parts = [], []
    for _, group in df.groupby("user_id", sort=False):
        n = len(group)
        n_test = max(1, int(round(n * test_frac)))
        train_parts.append(group.iloc[: n - n_test])
        test_parts.append(group.iloc[n - n_test :])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def build_user_items_map(train_df: pd.DataFrame) -> dict:
    """user_id -> set of movie_ids already rated in training data (used to
    exclude already-seen items from recommendations at serving time)."""
    return train_df.groupby("user_id")["movie_id"].apply(set).to_dict()


def build_relevant_items_map(test_df: pd.DataFrame, threshold: int = RELEVANCE_THRESHOLD) -> dict:
    """user_id -> set of movie_ids in the test set the user rated >= threshold
    (i.e. the ground-truth "liked" items used for Precision@K/Recall@K)."""
    relevant = test_df[test_df["rating"] >= threshold]
    return relevant.groupby("user_id")["movie_id"].apply(set).to_dict()


def build_movie_catalog(df: pd.DataFrame) -> pd.DataFrame:
    return df[["movie_id", "title"]].drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
