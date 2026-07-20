"""Lightweight pandas mirror of the PySpark cleaning/feature-engineering
logic in spark_jobs/preprocess.py.

This exists so the core data-transformation logic can be unit tested
quickly (see tests/test_preprocess.py) without spinning up a JVM/Spark
session in CI. The production pipeline still runs the real PySpark job
(spark_jobs/preprocess.py) via Airflow; this module is intentionally kept
in lockstep with the same cleaning rules.
"""
import pandas as pd

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    df = ratings.dropna()
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]
    df = df.drop_duplicates(subset=["user_id", "movie_id"])
    return df.reset_index(drop=True)


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    df = movies.dropna(subset=["movie_id", "title"]).drop_duplicates(subset=["movie_id"])
    df["release_year"] = df["title"].str.extract(r"\((\d{4})\)\s*$").astype("float")
    return df.reset_index(drop=True)


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    return users.dropna().drop_duplicates(subset=["user_id"]).reset_index(drop=True)


def add_genre_flags(movies: pd.DataFrame) -> pd.DataFrame:
    df = movies.copy()
    for genre in ALL_GENRES:
        col_name = "genre_" + genre.lower().replace("-", "_").replace("'", "")
        df[col_name] = df["genres"].str.contains(genre, regex=False).astype(int)
    return df


def add_rating_features(ratings: pd.DataFrame) -> pd.DataFrame:
    df = ratings.copy()
    df["rating_year"] = pd.to_datetime(df["timestamp"], unit="s").dt.year

    movie_stats = df.groupby("movie_id")["rating"].agg(
        movie_rating_count="count", movie_avg_rating="mean"
    )
    user_stats = df.groupby("user_id")["rating"].agg(
        user_rating_count="count", user_avg_rating="mean"
    )

    df = df.merge(movie_stats, on="movie_id", how="left")
    df = df.merge(user_stats, on="user_id", how="left")
    return df


def build_dataset(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    ratings = clean_ratings(ratings)
    movies = add_genre_flags(clean_movies(movies))
    users = clean_users(users)

    enriched = add_rating_features(ratings)
    full = enriched.merge(movies, on="movie_id", how="inner").merge(users, on="user_id", how="inner")

    full = full.fillna(
        {
            "movie_rating_count": 0,
            "movie_avg_rating": 0.0,
            "user_rating_count": 0,
            "user_avg_rating": 0.0,
            "release_year": 0,
        }
    )
    return full
