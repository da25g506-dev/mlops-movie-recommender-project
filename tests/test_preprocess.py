"""Unit tests for the pandas mirror of the Spark preprocessing pipeline
(src/data/preprocess_pandas.py), using tiny synthetic inputs shaped like
the real ml-1m .dat files."""
import pandas as pd

from src.data.preprocess_pandas import (
    add_genre_flags,
    build_dataset,
    clean_movies,
    clean_ratings,
    clean_users,
)


def sample_ratings():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "movie_id": [10, 20, 10, 20, 20],
            "rating": [5, 3, 6, 0, 4],  # rows with rating 6 and 0 are invalid
            "timestamp": [1000, 1001, 1002, 1003, 1003],  # last two rows are duplicates (user 2, movie 20)
        }
    )


def sample_movies():
    return pd.DataFrame(
        {
            "movie_id": [10, 20],
            "title": ["Movie Ten (1999)", "Movie Twenty (2005)"],
            "genres": ["Action|Comedy", "Drama"],
        }
    )


def sample_users():
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "gender": ["M", "F"],
            "age": [25, 35],
            "occupation": [1, 2],
            "zip_code": ["12345", "67890"],
        }
    )


def test_clean_ratings_drops_invalid_and_duplicate_rows():
    cleaned = clean_ratings(sample_ratings())
    assert cleaned["rating"].between(1, 5).all()
    assert not cleaned.duplicated(subset=["user_id", "movie_id"]).any()


def test_clean_movies_extracts_release_year():
    cleaned = clean_movies(sample_movies())
    assert cleaned.loc[cleaned["movie_id"] == 10, "release_year"].iloc[0] == 1999


def test_clean_users_drops_duplicates():
    users = pd.concat([sample_users(), sample_users().iloc[[0]]], ignore_index=True)
    cleaned = clean_users(users)
    assert len(cleaned) == 2


def test_add_genre_flags_creates_expected_columns():
    movies = clean_movies(sample_movies())
    flagged = add_genre_flags(movies)
    assert flagged.loc[flagged["movie_id"] == 10, "genre_action"].iloc[0] == 1
    assert flagged.loc[flagged["movie_id"] == 10, "genre_drama"].iloc[0] == 0
    assert flagged.loc[flagged["movie_id"] == 20, "genre_drama"].iloc[0] == 1


def test_build_dataset_merges_and_has_no_nulls():
    full = build_dataset(sample_ratings(), sample_movies(), sample_users())
    assert full.isnull().sum().sum() == 0
    assert "movie_avg_rating" in full.columns
    assert "user_avg_rating" in full.columns
    assert len(full) > 0
