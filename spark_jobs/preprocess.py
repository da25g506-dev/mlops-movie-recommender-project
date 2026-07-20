"""PySpark preprocessing job for the MovieLens ml-1m dataset.

Reads the raw `::`-delimited .dat files, cleans and merges them, engineers
features (genre one-hot flags, per-user / per-movie rating statistics,
rating year), and writes a single tidy Parquet dataset used downstream by
model training.

Why Spark: ratings.dat is a flat, tabular, append-only fact table joined
against two dimension tables (movies, users) with several groupBy
aggregations (per-user / per-movie stats) — a textbook batch-ETL shape.
Spark's DataFrame API expresses these joins/aggregations declaratively and
would transparently scale to the full ml-25m (or beyond) dataset on a real
cluster without a code change, which is why it's used here even though
ml-1m itself is small enough to fit in memory.
"""
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RATINGS_SCHEMA = StructType(
    [
        StructField("user_id", IntegerType(), False),
        StructField("movie_id", IntegerType(), False),
        StructField("rating", IntegerType(), False),
        StructField("timestamp", LongType(), False),
    ]
)

USERS_SCHEMA = StructType(
    [
        StructField("user_id", IntegerType(), False),
        StructField("gender", StringType(), False),
        StructField("age", IntegerType(), False),
        StructField("occupation", IntegerType(), False),
        StructField("zip_code", StringType(), False),
    ]
)

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def get_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("movielens-preprocess")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def read_dat(spark: SparkSession, filename: str, schema: StructType) -> DataFrame:
    path = str(RAW_DIR / filename)
    raw = spark.read.text(path)
    # ISO-8859-1 files come through as UTF-8-decoded text with mojibake for
    # non-ascii bytes; re-encode via a Python UDF-free round trip using the
    # dataframe's underlying RDD is expensive, so instead we read the byte
    # content already re-encoded to UTF-8 at the source file level (movies.dat
    # is handled specially below since it's the only file with free text).
    split_cols = F.split(raw["value"], "::")
    df = raw.select(
        *[split_cols.getItem(i).alias(f.name) for i, f in enumerate(schema.fields)]
    )
    for field in schema.fields:
        df = df.withColumn(field.name, df[field.name].cast(field.dataType))
    return df


def read_movies(spark: SparkSession) -> DataFrame:
    """movies.dat is ISO-8859-1 encoded and contains free-text titles with
    embedded commas, so it's parsed in Python and handed to Spark as a
    createDataFrame call rather than a naive text split."""
    path = RAW_DIR / "movies.dat"
    rows = []
    with open(path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            movie_id, title, genres = line.split("::")
            rows.append((int(movie_id), title, genres))
    schema = StructType(
        [
            StructField("movie_id", IntegerType(), False),
            StructField("title", StringType(), False),
            StructField("genres", StringType(), False),
        ]
    )
    return spark.createDataFrame(rows, schema=schema)


def clean_ratings(ratings: DataFrame) -> DataFrame:
    before = ratings.count()
    cleaned = (
        ratings.dropna()
        .filter((F.col("rating") >= 1) & (F.col("rating") <= 5))
        .dropDuplicates(["user_id", "movie_id"])
    )
    after = cleaned.count()
    logger.info("Ratings cleaned: %d -> %d rows (dropped %d)", before, after, before - after)
    return cleaned


def clean_movies(movies: DataFrame) -> DataFrame:
    before = movies.count()
    cleaned = movies.dropna(subset=["movie_id", "title"]).dropDuplicates(["movie_id"])
    # Extract release year from the title, e.g. "Toy Story (1995)"
    cleaned = cleaned.withColumn(
        "release_year",
        F.regexp_extract(F.col("title"), r"\((\d{4})\)\s*$", 1).cast(IntegerType()),
    )
    after = cleaned.count()
    logger.info("Movies cleaned: %d -> %d rows (dropped %d)", before, after, before - after)
    return cleaned


def clean_users(users: DataFrame) -> DataFrame:
    before = users.count()
    cleaned = users.dropna().dropDuplicates(["user_id"])
    after = cleaned.count()
    logger.info("Users cleaned: %d -> %d rows (dropped %d)", before, after, before - after)
    return cleaned


def add_genre_flags(movies: DataFrame) -> DataFrame:
    df = movies
    for genre in ALL_GENRES:
        col_name = "genre_" + genre.lower().replace("-", "_").replace("'", "")
        df = df.withColumn(col_name, F.array_contains(F.split(F.col("genres"), r"\|"), genre).cast(IntegerType()))
    return df


def add_rating_features(ratings: DataFrame) -> DataFrame:
    df = ratings.withColumn("rating_year", F.year(F.to_timestamp(F.col("timestamp"))))

    movie_stats = df.groupBy("movie_id").agg(
        F.count("rating").alias("movie_rating_count"),
        F.avg("rating").alias("movie_avg_rating"),
    )
    user_stats = df.groupBy("user_id").agg(
        F.count("rating").alias("user_rating_count"),
        F.avg("rating").alias("user_avg_rating"),
    )

    df = df.join(movie_stats, on="movie_id", how="left")
    df = df.join(user_stats, on="user_id", how="left")
    return df


def build_dataset(spark: SparkSession) -> DataFrame:
    ratings = clean_ratings(read_dat(spark, "ratings.dat", RATINGS_SCHEMA))
    movies = add_genre_flags(clean_movies(read_movies(spark)))
    users = clean_users(read_dat(spark, "users.dat", USERS_SCHEMA))

    enriched = add_rating_features(ratings)

    full = (
        enriched.join(movies, on="movie_id", how="inner")
        .join(users, on="user_id", how="inner")
    )

    # Guard against any nulls introduced by the joins/aggregations before
    # this feature set is consumed by model training.
    full = full.na.fill(
        {
            "movie_rating_count": 0,
            "movie_avg_rating": 0.0,
            "user_rating_count": 0,
            "user_avg_rating": 0.0,
            "release_year": 0,
        }
    )

    logger.info("Final merged dataset: %d rows, %d columns", full.count(), len(full.columns))
    return full


def main() -> None:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        full = build_dataset(spark)
        out_path = str(PROCESSED_DIR / "ratings_features.parquet")
        full.coalesce(1).write.mode("overwrite").parquet(out_path)
        logger.info("Wrote processed dataset to %s", out_path)

        # Also persist a flat CSV export for tools that don't read Parquet
        # (e.g. quick pandas inspection, Evidently reference data).
        csv_out = str(PROCESSED_DIR / "ratings_features.csv")
        full.toPandas().to_csv(csv_out, index=False)
        logger.info("Wrote CSV export to %s", csv_out)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
