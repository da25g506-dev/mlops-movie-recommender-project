"""Bayesian Personalized Ranking recommender, using the `implicit` library.

Unlike ALS (which minimizes reconstruction error on the confidence-weighted
rating matrix), BPR directly optimizes pairwise ranking - for each user it
learns to score observed items above unobserved ones - which is a closer
match to what Precision@K/Recall@K actually measure. Provided as a fourth
model family alongside the popularity baseline, SVD, and ALS, to compare a
ranking-first objective against a reconstruction-first one on the same
implicit-feedback task.

Uses raw rating values as confidence weights, same as ALSModel (see its
docstring: BM25 re-weighting was tried and measured worse on a held-out
split of this dataset, so it isn't used here either).
"""
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.bpr import BayesianPersonalizedRanking


class BPRModel:
    def __init__(self, factors: int = 30, regularization: float = 0.1, iterations: int = 200, learning_rate: float = 0.005, random_state: int = 42):
        self.params = dict(factors=factors, regularization=regularization, iterations=iterations, learning_rate=learning_rate)
        self.model = BayesianPersonalizedRanking(random_state=random_state, **self.params)
        self.user_to_idx_: Dict[int, int] = {}
        self.idx_to_movie_: Dict[int, int] = {}
        self.movie_to_idx_: Dict[int, int] = {}
        self.user_items_csr_: sp.csr_matrix | None = None
        self.global_mean_: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "BPRModel":
        self.global_mean_ = float(train_df["rating"].mean())

        users = train_df["user_id"].unique()
        movies = train_df["movie_id"].unique()
        self.user_to_idx_ = {u: i for i, u in enumerate(users)}
        self.movie_to_idx_ = {m: i for i, m in enumerate(movies)}
        self.idx_to_movie_ = {i: m for m, i in self.movie_to_idx_.items()}

        row = train_df["user_id"].map(self.user_to_idx_).to_numpy()
        col = train_df["movie_id"].map(self.movie_to_idx_).to_numpy()
        data = train_df["rating"].to_numpy(dtype=np.float32)

        self.user_items_csr_ = sp.csr_matrix((data, (row, col)), shape=(len(users), len(movies)))
        self.model.fit(self.user_items_csr_)
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """BPR optimizes pairwise ranking, not explicit rating values, so
        raw dot-product scores are not on the 1-5 rating scale. RMSE/MAE
        are therefore not meaningful for this model and it is evaluated on
        Precision@K/Recall@K only; this method exists so the same
        model interface can still be probed, e.g. for relative ranking."""
        if user_id not in self.user_to_idx_ or movie_id not in self.movie_to_idx_:
            return self.global_mean_
        u_idx = self.user_to_idx_[user_id]
        m_idx = self.movie_to_idx_[movie_id]
        score = float(self.model.user_factors[u_idx] @ self.model.item_factors[m_idx])
        return score

    def recommend(self, user_id: int, k: int, exclude_items: set) -> List[int]:
        if user_id not in self.user_to_idx_:
            return []
        u_idx = self.user_to_idx_[user_id]
        exclude_idx = {self.movie_to_idx_[m] for m in exclude_items if m in self.movie_to_idx_}

        ids, _scores = self.model.recommend(
            u_idx,
            self.user_items_csr_[u_idx],
            N=k + len(exclude_idx),
            filter_already_liked_items=False,
        )
        recs = [self.idx_to_movie_[i] for i in ids if i not in exclude_idx]
        return recs[:k]

    def recommend_batch(self, user_ids: List[int], k: int, user_items_map: Dict[int, set]) -> Dict[int, List[int]]:
        return {u: self.recommend(u, k, user_items_map.get(u, set())) for u in user_ids}
