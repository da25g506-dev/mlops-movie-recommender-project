"""Popularity baseline recommender.

Predicts every user's rating for a movie as that movie's global average
rating (a classic non-personalized cold-start baseline), and recommends
the highest-average-rated movies (with a minimum rating-count floor to
avoid recommending obscure one-rating movies) to every user.
"""
from typing import Dict, List

import pandas as pd


class PopularityModel:
    def __init__(self, min_ratings: int = 20):
        self.min_ratings = min_ratings
        self.movie_avg_rating_: pd.Series | None = None
        self.global_mean_: float = 0.0
        self.ranked_movies_: List[int] = []

    def fit(self, train_df: pd.DataFrame) -> "PopularityModel":
        stats = train_df.groupby("movie_id")["rating"].agg(["mean", "count"])
        self.movie_avg_rating_ = stats["mean"]
        self.global_mean_ = float(train_df["rating"].mean())

        eligible = stats[stats["count"] >= self.min_ratings]
        self.ranked_movies_ = eligible.sort_values("mean", ascending=False).index.tolist()
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if self.movie_avg_rating_ is not None and movie_id in self.movie_avg_rating_.index:
            return float(self.movie_avg_rating_.loc[movie_id])
        return self.global_mean_

    def recommend(self, user_id: int, k: int, exclude_items: set) -> List[int]:
        recs = [m for m in self.ranked_movies_ if m not in exclude_items]
        return recs[:k]

    def recommend_batch(self, user_ids: List[int], k: int, user_items_map: Dict[int, set]) -> Dict[int, List[int]]:
        return {u: self.recommend(u, k, user_items_map.get(u, set())) for u in user_ids}
