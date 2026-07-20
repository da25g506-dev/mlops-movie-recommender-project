"""SVD-based collaborative filtering model, using scikit-surprise.

Classic matrix-factorization collaborative filtering: learns latent
user/item factor vectors from the observed rating matrix and predicts
unseen ratings as their dot product (plus bias terms). Chosen as the
"classic CF" reference point alongside the popularity baseline and the
implicit-feedback ALS model.
"""
from typing import Dict, List

import pandas as pd
from surprise import SVD, Dataset, Reader


class SVDModel:
    def __init__(self, n_factors: int = 50, n_epochs: int = 20, lr_all: float = 0.005, reg_all: float = 0.02, random_state: int = 42):
        self.params = dict(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, random_state=random_state)
        self.model = SVD(**self.params)
        self.all_movie_ids_: List[int] = []

    def fit(self, train_df: pd.DataFrame) -> "SVDModel":
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        self.all_movie_ids_ = train_df["movie_id"].unique().tolist()
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        return float(self.model.predict(user_id, movie_id).est)

    def recommend(self, user_id: int, k: int, exclude_items: set) -> List[int]:
        candidates = [m for m in self.all_movie_ids_ if m not in exclude_items]
        scored = [(m, self.predict_rating(user_id, m)) for m in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:k]]

    def recommend_batch(self, user_ids: List[int], k: int, user_items_map: Dict[int, set]) -> Dict[int, List[int]]:
        return {u: self.recommend(u, k, user_items_map.get(u, set())) for u in user_ids}
