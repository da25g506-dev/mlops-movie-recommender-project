"""Unit tests for the three recommender models and shared metrics, using a
small synthetic ratings dataset so tests run in milliseconds without
touching the full ml-1m data or a live MLflow server."""
import pandas as pd
import pytest

from src.models.als_model import ALSModel
from src.models.baseline_popularity import PopularityModel
from src.models.data_utils import build_relevant_items_map, build_user_items_map, train_test_split_by_user
from src.models.metrics import mae, precision_recall_at_k, rmse
from src.models.svd_model import SVDModel


@pytest.fixture
def synthetic_ratings():
    rows = []
    timestamp = 1_000_000_000
    for user_id in range(1, 11):
        for movie_id in range(1, 21):
            if (user_id + movie_id) % 3 != 0:
                rows.append(
                    {
                        "user_id": user_id,
                        "movie_id": movie_id,
                        "rating": ((user_id + movie_id) % 5) + 1,
                        "timestamp": timestamp,
                        "title": f"Movie {movie_id}",
                    }
                )
                timestamp += 1
    return pd.DataFrame(rows)


def test_train_test_split_by_user_preserves_all_rows(synthetic_ratings):
    train_df, test_df = train_test_split_by_user(synthetic_ratings, test_frac=0.2)
    assert len(train_df) + len(test_df) == len(synthetic_ratings)
    assert set(train_df["user_id"]) <= set(synthetic_ratings["user_id"])
    assert set(test_df["user_id"]) <= set(synthetic_ratings["user_id"])


def test_rmse_and_mae_basic():
    y_true = [3, 4, 5]
    y_pred = [3, 4, 5]
    assert rmse(y_true, y_pred) == 0.0
    assert mae(y_true, y_pred) == 0.0
    assert rmse([1, 2], [3, 4]) == pytest.approx(2.0)


def test_precision_recall_at_k():
    recs = {1: [10, 20, 30]}
    relevant = {1: {20, 30, 40}}
    result = precision_recall_at_k(recs, relevant, k=3)
    assert result["precision_at_3"] == pytest.approx(2 / 3)
    assert result["recall_at_3"] == pytest.approx(2 / 3)


def test_popularity_model_fit_and_recommend(synthetic_ratings):
    train_df, test_df = train_test_split_by_user(synthetic_ratings, test_frac=0.3)
    model = PopularityModel(min_ratings=1).fit(train_df)

    recs = model.recommend(user_id=1, k=5, exclude_items=set())
    assert len(recs) <= 5
    assert isinstance(model.predict_rating(1, train_df["movie_id"].iloc[0]), float)


def test_svd_model_fit_and_recommend(synthetic_ratings):
    train_df, test_df = train_test_split_by_user(synthetic_ratings, test_frac=0.3)
    model = SVDModel(n_factors=4, n_epochs=5).fit(train_df)

    pred = model.predict_rating(1, train_df["movie_id"].iloc[0])
    assert 1.0 <= pred <= 5.0

    recs = model.recommend(user_id=1, k=5, exclude_items=set())
    assert len(recs) <= 5


def test_als_model_fit_and_recommend(synthetic_ratings):
    train_df, test_df = train_test_split_by_user(synthetic_ratings, test_frac=0.3)
    model = ALSModel(factors=4, iterations=3).fit(train_df)

    user_items_map = build_user_items_map(train_df)
    relevant_items_map = build_relevant_items_map(test_df)
    test_users = list(relevant_items_map.keys())

    recs = model.recommend_batch(test_users, k=5, user_items_map=user_items_map)
    assert set(recs.keys()) == set(test_users)
    for user_id, rec_list in recs.items():
        assert len(rec_list) <= 5
        assert not (set(rec_list) & user_items_map.get(user_id, set()))
