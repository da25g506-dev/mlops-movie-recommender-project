"""ALS-based matrix factorization recommender, using the `implicit` library.

Implicit ALS treats ratings as confidence-weighted implicit feedback
signals (rather than modeling explicit rating values directly), which is
the standard approach for top-K recommendation quality. Provided as the
third, architecturally distinct model alongside the popularity baseline
and the explicit-rating SVD model.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class ALSModel:
    def __init__(self, factors: int = 50, regularization: float = 0.01, iterations: int = 15, random_state: int = 42):
        self.params = dict(factors=factors, regularization=regularization, iterations=iterations)
        self.model = AlternatingLeastSquares(random_state=random_state, **self.params)
        self.user_to_idx_: Dict[int, int] = {}
        self.idx_to_movie_: Dict[int, int] = {}
        self.movie_to_idx_: Dict[int, int] = {}
        self.user_items_csr_: sp.csr_matrix | None = None
        self.global_mean_: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "ALSModel":
        self.global_mean_ = float(train_df["rating"].mean())

        users = train_df["user_id"].unique()
        movies = train_df["movie_id"].unique()
        self.user_to_idx_ = {u: i for i, u in enumerate(users)}
        self.movie_to_idx_ = {m: i for i, m in enumerate(movies)}
        self.idx_to_movie_ = {i: m for m, i in self.movie_to_idx_.items()}

        row = train_df["user_id"].map(self.user_to_idx_).to_numpy()
        col = train_df["movie_id"].map(self.movie_to_idx_).to_numpy()
        # Rating value itself used as the implicit "confidence" weight.
        data = train_df["rating"].to_numpy(dtype=np.float32)

        self.user_items_csr_ = sp.csr_matrix((data, (row, col)), shape=(len(users), len(movies)))
        self.model.fit(self.user_items_csr_)
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """ALS optimizes implicit affinity, not explicit rating values, so
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
