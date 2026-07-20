# 🎬 Movie Recommender System — End-to-End MLOps Project

An end-to-end, production-style MLOps pipeline for a **movie recommendation system**, built on the [MovieLens ml-1m](https://grouplens.org/datasets/movielens/1m/) dataset (1M ratings, 6,040 users, 3,900 movies).

This project satisfies the MLOps Course End-Term Project requirements: an automated data pipeline (Airflow + Spark), data processing, multi-model development with experiment tracking (MLflow), dataset/model versioning (DVC), containerized deployment (FastAPI + Docker), monitoring (Prometheus + Grafana + Evidently drift detection), and CI/CD (GitHub Actions).

See [REPORT.md](REPORT.md) for the full technical report, and [docs/PROGRESS.md](docs/PROGRESS.md) for the build stage log.

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipelines & Services](#running-the-pipelines--services)
- [API Usage](#api-usage)
- [Docker Commands](#docker-commands)
- [Monitoring](#monitoring)
- [CI/CD](#cicd)
- [Dependencies](#dependencies)

## Project Overview

Given a user ID, the system recommends the top-K movies that user is most likely to enjoy, trained on the MovieLens ml-1m dataset. Three model families are trained and tracked side by side:

| Model | Library | Type | RMSE | MAE | Precision@10 | Recall@10 |
|---|---|---|---|---|---|---|
| Popularity baseline | pandas | non-personalized | 0.989 | 0.787 | 0.029 | 0.016 |
| SVD | scikit-surprise | explicit-feedback matrix factorization | 0.887 | 0.695 | 0.051 | 0.028 |
| **ALS** (selected) | implicit | implicit-feedback matrix factorization | n/a* | n/a* | **0.085** | **0.082** |

\* ALS is trained on implicit confidence weights rather than explicit 1–5 ratings, so rating-error metrics don't apply to it; it's evaluated purely on ranking quality (Precision@K / Recall@K), which is also the metric used to pick a production model. See [REPORT.md](REPORT.md) for the full methodology and discussion.

`src/models/evaluate.py` compares the most recent run of each model family by `recall_at_10`, promotes the winner to the `Production` stage of the `movie-recommender-prod` registered model in MLflow, and exports its artifacts to `models/production_model/`, which is then DVC-tracked and pushed by the `dvc_commit_model` Airflow task — so the artifact MLflow serves and the one DVC versions are the same, both traceable back to the exact MLflow run that produced them. ALS currently wins and is what the API serves.

## System Architecture

```
MovieLens ml-1m (raw .dat files)
        │
        ▼
┌────────────────────────────────── Airflow (movie_recommender_pipeline, @daily) ───────────────────────────────────┐
│  download_data → spark_preprocess → dvc_commit_processed → train_models → evaluate_and_register → dvc_commit_model│
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │                       │                                  │                  │                  │
        ▼                       ▼                                  ▼                  ▼                  ▼
  data/raw/*.dat      data/processed/*.parquet,csv         MLflow tracking      MLflow Model Registry   models/production_model/
  (DVC-tracked)             (DVC-tracked)                 (3 runs, metrics,      (movie-recommender-prod,  (DVC-tracked export
                                                            params, artifacts)    stage=Production)          of the Production model)
                                                                                          │
                                                                                          ▼
                                                                          FastAPI service (Docker, port 8000)
                                                                          GET /recommend/{user_id}?k=10
                                                                          GET /health · GET /metrics
                                                                          GET /drift-metrics
                                                                          → prediction_logs/predictions.jsonl
                                                                                          │
                        ┌─────────────────────────────────────────────────────────────────┴───────────┐
                        ▼                                                                             ▼
        Airflow (drift_monitoring, every 30 min)                                  Prometheus (port 9090)
        monitoring/drift_detection.py (Evidently)                                 scrapes /metrics + /drift-metrics
        writes drift_reports/*.html + monitoring/drift_status.json                        │
                                                                                            ▼
                                                                                Grafana (port 3000)
                                                                                request rate, latency, errors,
                                                                                drift flag/share/samples
```

Every service (Postgres, Airflow webserver + scheduler, MLflow, FastAPI, Prometheus, Grafana) is defined in a single [`docker-compose.yml`](docker-compose.yml).

**Why Spark for preprocessing?** ml-1m is small enough to fit in memory, but the transformations (per-user/per-movie aggregates, genre one-hot encoding, multi-table joins) are the same shape as batch feature-engineering jobs on much larger rating logs — Spark demonstrates the distributed-ETL pattern this pipeline would scale to, while a pandas mirror (`src/data/preprocess_pandas.py`) keeps unit tests fast and JVM-free.

**Why three different model families?** Popularity (non-personalized baseline), SVD (classic explicit-feedback CF via scikit-surprise), and ALS (implicit-feedback matrix factorization via `implicit`) represent three genuinely different modeling approaches, evaluated consistently on Precision@10/Recall@10 (and RMSE/MAE where the model produces a rating estimate).

## Repository Structure

```
movie-recommender-mlops/
├── .github/workflows/ci-cd.yml       # Lint → test → build → push-to-GHCR pipeline
├── .flake8                            # Shared lint config
├── airflow/dags/
│   ├── movie_recommender_pipeline.py  # download → preprocess → DVC → train → evaluate/register (@daily)
│   └── drift_monitoring.py            # standalone drift check (every 30 min)
├── spark_jobs/preprocess.py           # PySpark ETL: clean, dedupe, merge, feature-engineer
├── src/
│   ├── data/
│   │   ├── download.py                # fetch + checksum-verify ml-1m
│   │   └── preprocess_pandas.py       # pandas mirror of the Spark job (used by unit tests)
│   ├── models/
│   │   ├── baseline_popularity.py     # non-personalized baseline
│   │   ├── svd_model.py               # scikit-surprise SVD
│   │   ├── als_model.py               # implicit ALS
│   │   ├── recommender_pyfunc.py      # MLflow pyfunc wrapper (shared serving interface)
│   │   ├── data_utils.py              # load/split/catalog helpers
│   │   ├── metrics.py                 # RMSE, MAE, Precision@K, Recall@K
│   │   ├── train.py                   # trains + logs all 3 models to MLflow
│   │   └── evaluate.py                # compares runs, registers + promotes the best model
│   └── serving/
│       ├── model_loader.py            # loads the Production model + movie catalog
│       └── app.py                     # FastAPI app: /recommend, /health, /metrics, /drift-metrics
├── monitoring/
│   ├── prometheus.yml                 # scrape config
│   ├── drift_detection.py             # Evidently DataDriftPreset vs. live traffic
│   └── grafana/                       # provisioned datasource + dashboard (as code)
├── scripts/dvc_commit.py              # `dvc add` + `dvc push` helper used by Airflow
├── tests/                             # pytest: preprocessing, models, API, drift detection
├── data/raw/, data/processed/         # DVC-tracked (git only holds the .dvc pointer files)
├── models/production_model/            # DVC-tracked export of the current Production model
├── docker-compose.yml                 # Postgres, Airflow, MLflow, API, Prometheus, Grafana
├── Dockerfile.airflow / .mlflow / .api
├── requirements.txt                   # full pipeline environment (Airflow image)
├── requirements-api.txt               # minimal serving-only environment (API image)
├── REPORT.md                          # technical report
└── docs/PROGRESS.md                   # build-stage log
```

## Setup & Installation

**Prerequisites:** Docker + Docker Compose, ~6 GB free disk (Docker images + MLflow/Airflow/Postgres volumes), internet access (to download the dataset on first run).

```bash
git clone git@github.com:da25g506-dev/mlops-movie-recommender-project.git movie-recommender-mlops
cd movie-recommender-mlops

# (Optional) point DVC at your own local remote instead of the default path
# baked into .dvc/config, then `dvc pull` to fetch already-versioned artifacts.

docker compose up -d --build
```

This builds and starts 7 containers: `postgres`, `airflow-webserver`, `airflow-scheduler`, `mlflow`, `api`, `prometheus`, `grafana`. First boot takes a few minutes while images build and Airflow's metadata DB migrates.

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI | http://localhost:5000 | — |
| FastAPI docs | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

## Running the Pipelines & Services

**1. Run the training pipeline** (downloads data, preprocesses with Spark, versions with DVC, trains 3 models, registers the best one):

```bash
# Airflow UI: unpause + trigger "movie_recommender_pipeline", or via CLI:
docker exec -it movie-recommender-mlops-airflow-scheduler-1 airflow dags trigger movie_recommender_pipeline
```

Watch task logs in the Airflow UI. On success, check the MLflow UI (`movie-recommender` experiment) for 3 runs with logged params/metrics/artifacts, and the **Models** tab for `movie-recommender-prod` promoted to `Production`.

**2. Restart the API** to pick up a newly registered model (it loads the Production model once at startup):

```bash
docker compose restart api
```

**3. Run the drift check** (compares live `/recommend` traffic against the training reference distribution — needs ≥30 logged predictions first):

```bash
docker exec -it movie-recommender-mlops-airflow-scheduler-1 airflow dags trigger drift_monitoring
# or directly:
docker exec -it movie-recommender-mlops-airflow-scheduler-1 python3 monitoring/drift_detection.py
```

Runs automatically every 30 minutes once the `drift_monitoring` DAG is unpaused (it is, by default — `AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=false`).

## API Usage

```bash
# Get 5 recommendations for user 1
curl "http://localhost:8000/recommend/1?k=5"
# {"user_id":1,"k":5,"recommendations":[{"movie_id":1961,"title":"Rain Man (1988)"}, ...]}

# Health check
curl "http://localhost:8000/health"
# {"status":"ok"}

# Prometheus metrics (request rate, latency histograms, etc.)
curl "http://localhost:8000/metrics"

# Latest drift-check result, as Prometheus gauges
curl "http://localhost:8000/drift-metrics"
```

`k` must be between 1 and 100 (default 10); invalid values return `400`. Every `/recommend` call is appended to `prediction_logs/predictions.jsonl` for drift monitoring.

## Docker Commands

```bash
docker compose up -d --build      # build + start everything
docker compose ps                 # check container health
docker compose logs -f api        # tail a specific service's logs
docker compose build api          # rebuild just the API image after a code change
docker compose restart api        # pick up a newly registered model or config change
docker compose down               # stop everything (add -v to also drop named volumes)
```

## Monitoring

- **Prometheus** (`monitoring/prometheus.yml`) scrapes two targets on the `api` container: `/metrics` (auto-instrumented request rate, latency percentiles, error rate via `prometheus-fastapi-instrumentator`) and `/drift-metrics` (custom gauges: `recommender_dataset_drift`, `recommender_drift_share`, `recommender_drift_samples`).
- **Grafana** (`monitoring/grafana/`) is provisioned entirely as code — datasource and a 7-panel dashboard ("Movie Recommender API") are loaded automatically on container start, no manual UI setup required.
- **Drift detection** (`monitoring/drift_detection.py`) uses Evidently's `DataDriftPreset` to compare the per-movie feature distribution of *recommended* movies (from live traffic) against the *full catalog* (the training reference). It writes an HTML report to `drift_reports/` and a JSON status file the API exposes to Prometheus. Runs via the `drift_monitoring` Airflow DAG every 30 minutes, or on demand.

## CI/CD

[`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml) runs on every push/PR:

1. **lint-and-test**: install `requirements.txt`, `flake8` lint, run the full `pytest` suite (preprocessing, all 3 models, the FastAPI serving layer, drift detection — 24 tests).
2. **docker**: fetch the raw dataset (needed to bake `movies.dat` into the API image), build `Dockerfile.api`. On pushes to `main`, additionally push the image to `ghcr.io/da25g506-dev/mlops-movie-recommender-project` tagged `:latest` and `:<commit-sha>`, using the repo's built-in `GITHUB_TOKEN` (no extra secret needed).

## Dependencies

- [`requirements.txt`](requirements.txt) — full pipeline environment (Spark, DVC, MLflow, all 3 model libraries, Evidently, testing/lint tools). Used by the Airflow image and by CI.
- [`requirements-api.txt`](requirements-api.txt) — minimal serving-only environment (no PySpark/DVC/Evidently/pytest) used by the always-on FastAPI image.
