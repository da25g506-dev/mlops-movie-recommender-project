# Technical Report: Movie Recommender System — An End-to-End MLOps Pipeline

**Course:** MLOps Course End-Term Project
**Repository:** `da25g506-dev/mlops-movie-recommender-project`

## 1. Problem Statement

Recommendation systems are one of the highest-leverage applications of ML in production: they sit directly in a user-facing product path, are retrained continuously as user behavior shifts, and require the full MLOps toolchain to operate reliably — versioned data, reproducible training, tracked experiments, automated deployment, and live monitoring for degradation. This project builds a complete, working system around that lifecycle rather than a single notebook model, using movie recommendation as the concrete task: given a user ID, return the top-K movies that user is likely to rate highly.

The goal is not to maximize offline accuracy on a leaderboard, but to demonstrate — with a real, runnable system — every stage an ML model goes through before and after it reaches production: automated data ingestion and feature engineering, multi-model experimentation with tracked comparisons, versioned datasets and models, containerized serving, and monitoring that can detect when the deployed model's behavior starts to drift from what it saw in training.

## 2. Dataset

**MovieLens ml-1m** (GroupLens Research): 1,000,209 ratings from 6,040 users on 3,900 movies, collected 2000–2003. Distributed as three `::`-delimited flat files:

- `ratings.dat` — `user_id::movie_id::rating::timestamp` (rating ∈ {1..5})
- `movies.dat` — `movie_id::title (year)::genre1|genre2|...`
- `users.dat` — `user_id::gender::age::occupation::zip_code`

Chosen because it's large enough to justify a Spark-based ETL step and to make matrix-factorization models meaningful, while still small enough to train end-to-end on a single machine in seconds — appropriate for a course project that needs to be run repeatedly and demoed live, not queued on a cluster.

`src/data/download.py` fetches the official archive from `https://files.grouplens.org/datasets/movielens/ml-1m.zip`, verifies its MD5 checksum (`c4d9eecfca2ab87c1945afe126590906`) before extracting, and validates that all three expected files are present and non-empty — a deliberate fail-fast step so a corrupted or partial download doesn't silently propagate into training.

## 3. System Architecture

```
MovieLens ml-1m (raw .dat files)
        │
        ▼
Airflow DAG "movie_recommender_pipeline" (@daily)
  download_data → spark_preprocess → dvc_commit_processed → train_models → evaluate_and_register → dvc_commit_model
        │                  │                    │                  │              │                      │
        ▼                  ▼                    ▼                  ▼              ▼                      ▼
   data/raw/*.dat   data/processed/*.parquet   DVC remote      MLflow runs   MLflow Model Registry   models/production_model/
                                                                              (movie-recommender-prod,  (DVC-tracked, pushed
                                                                               stage = Production)        to the same remote)
                                                                                      │
                                                                                      ▼
                                                                      FastAPI service (Docker container)
                                                                      /recommend  /health  /metrics  /drift-metrics
                                                                                      │
                                                                                      ▼
                                                                  Kafka topic "recommendation-events"
                                                                      (kafka-ui for visual inspection)
                                                                                      │
                                        ┌─────────────────────────────────────────────┴──────────────┐
                                        ▼                                                             │
                        Airflow DAG "drift_monitoring" (every 30 min)                                 │
                        consume_kafka_events → prediction_logs/predictions.jsonl                      │
                                        │                                                              │
                                        ▼                                                              │
                        aggregate_recommendations (Beam DirectRunner)                                 │
                        → monitoring/recommendation_frequency.json                                    │
                                        │                                                              │
                                        ▼                                                              ▼
                        check_drift (Evidently vs. training reference)                    Prometheus (scrapes API)
                        → drift_reports/*.html, monitoring/drift_status.json                          │
                                                                                                        ▼
                                                                                              Grafana dashboards
```

Every service — Postgres (Airflow's metadata DB), Airflow webserver + scheduler, the MLflow tracking server, the FastAPI app, Zookeeper + Kafka + kafka-ui, Prometheus, and Grafana — runs as a container defined in a single `docker-compose.yml`, so the entire system comes up with `docker compose up -d --build` and is reproducible on any machine with Docker.

**Design choices and their rationale:**

- **Airflow + Postgres, LocalExecutor.** A single webserver + scheduler pair backed by Postgres demonstrates real DAG scheduling, retries, and task dependency management without the operational overhead of CeleryExecutor/Redis, which buys nothing at this data volume.
- **MLflow with HTTP-proxied artifact storage.** The tracking server serves artifacts over its own HTTP API (`--serve-artifacts --artifacts-destination`) rather than requiring every client to have direct filesystem access to its storage volume. This was a deliberate fix after the initial local-filesystem artifact root caused permission errors from the Airflow container (see §10, Challenges) — and it's the architecturally correct pattern for a tracking server with remote clients.
- **DVC with a local-directory remote.** Demonstrates real `dvc add`/`dvc push`/`dvc pull` versioning and a genuine separation between git (code + small pointer files) and DVC (large data/model binaries) without requiring cloud storage credentials.
- **Two separate Airflow DAGs.** Training (`movie_recommender_pipeline`, daily) and drift monitoring (`drift_monitoring`, every 30 minutes) run on independent schedules because they have fundamentally different cadences and failure domains — a drift-check failure shouldn't block or be coupled to a training run, and vice versa.
- **Kafka between serving and monitoring.** The FastAPI service publishes to a `recommendation-events` topic instead of writing `prediction_logs/predictions.jsonl` directly. This decouples the always-on serving path from the batch monitoring pipeline: the API keeps working even if the monitoring/consumer side is down, and it would let multiple API replicas publish to the same topic without a shared filesystem — a genuine architectural reason to use a broker rather than a direct file write, not a checkbox integration.
- **Beam for recommendation-concentration aggregation.** `beam_jobs/aggregate_recommendations.py` is a small, portable batch job (DirectRunner, in-process, no cluster) computing per-movie recommendation frequency and a `top_movie_share` concentration signal — deliberately a different job shape from the Spark preprocessing ETL, to demonstrate Beam's programming model on a task suited to it (a lightweight, portable batch aggregation) rather than duplicating Spark's role.

## 4. Data Pipeline

**Cleaning** (`spark_jobs/preprocess.py`, mirrored in pandas for tests by `src/data/preprocess_pandas.py`):
- Ratings: drop nulls, keep only `1 <= rating <= 5`, drop duplicate `(user_id, movie_id)` pairs.
- Movies: drop nulls on `movie_id`/`title`, drop duplicate `movie_id`, extract `release_year` from the title via regex.
- Users: drop nulls, drop duplicate `user_id`.

**Merging:** inner join ratings → movies → users on `movie_id` / `user_id`, producing one row per rating with the full context attached.

**Feature engineering:**
- 18 one-hot genre flags (`genre_action` … `genre_western`) parsed from the `|`-delimited genre string.
- Per-movie aggregates: `movie_rating_count`, `movie_avg_rating`.
- Per-user aggregates: `user_rating_count`, `user_avg_rating`.
- `rating_year` extracted from the Unix timestamp.

**Output:** `data/processed/ratings_features.parquet` (and a `.csv` twin for easy inspection), containing the joined, cleaned, feature-engineered table used both for model training and as the drift-detection reference distribution.

**Why Spark:** the transformations are the textbook shape of batch feature engineering on a ratings log — multi-table joins, groupby aggregates, one-hot encoding — the same operations that wouldn't fit in memory at real-world scale (tens of millions of users). Running them through Spark here demonstrates the distributed-processing pattern even though ml-1m itself comfortably fits in memory; `spark_jobs/preprocess.py` would scale to a cluster with no code changes beyond the master URL. A pandas-only mirror of the same logic exists purely so unit tests can verify the transformation rules without spinning up a JVM in CI.

**Versioning:** `scripts/dvc_commit.py` (invoked as an Airflow task) runs `dvc add` on the raw `.dat` files and the processed parquet/csv, then `dvc push`s to the configured local-directory DVC remote — so every pipeline run leaves a versioned, retrievable snapshot of exactly what data a given model was trained on.

## 5. Model Development

Four model families are trained on the same 80/20 per-user temporal split (`train_test_split_by_user`: each user's most recent 20% of ratings, by timestamp, held out — chosen over a random split because it evaluates the more realistic task of predicting future behavior from past behavior, not interpolating within a user's history):

| Model | Library | Approach | Key hyperparameters |
|---|---|---|---|
| Popularity baseline | pandas | Global mean rating per movie (non-personalized) | `min_ratings=20` |
| SVD | scikit-surprise | Explicit-feedback matrix factorization | `n_factors=50`, `n_epochs=20`, `lr_all=0.005`, `reg_all=0.02` |
| ALS | `implicit` | Implicit-feedback (confidence-weighted) matrix factorization, reconstruction-error objective | `factors=60`, `regularization=0.01`, `iterations=8` |
| BPR | `implicit` | Implicit-feedback matrix factorization, pairwise-ranking objective | `factors=30`, `regularization=0.1`, `iterations=200`, `learning_rate=0.005` |

All four are wrapped in a common `RecommenderPyfunc` MLflow pyfunc interface (`src/models/recommender_pyfunc.py`) so the serving layer can call `model.predict(pd.DataFrame({"user_id": [...], "k": [...]}))` identically regardless of which family won.

**Evaluation metrics** (`src/models/metrics.py`): RMSE and MAE on held-out ratings (where the model produces a rating estimate — ALS and BPR don't, since they fit implicit factors rather than modeling the 1–5 scale directly), plus Precision@10 and Recall@10 against a relevance threshold of rating ≥ 4, which is the metric all four models can be compared on regardless of whether they predict explicit ratings.

**On BM25 confidence weighting:** the standard refinement for turning explicit ratings into implicit "confidence" for ALS/BPR is BM25 re-weighting (`implicit.nearest_neighbours.bm25_weight`), which downweights globally-popular items. It was implemented and evaluated on this exact train/test split via grid search over `factors`/`regularization`/`iterations`/`learning_rate`, both with and without BM25. The result was consistent and reproducible (deterministic given `random_state=42`): BM25 weighting *reduced* Precision@10/Recall@10 for both ALS and BPR relative to using raw 1–5 rating values directly as confidence weights. At this dataset's scale (6,040 users × 3,900 movies), BM25's popularity-downweighting appears to remove signal that's actually useful rather than noise — so both models use raw ratings as confidence weights, and the hyperparameters below are the best configuration found under that scheme, not the textbook default.

**Results** (actual run from the `movie-recommender` MLflow experiment, via a live Airflow-triggered `movie_recommender_pipeline` run):

| Model | RMSE | MAE | Precision@10 | Recall@10 |
|---|---|---|---|---|
| Popularity baseline | 0.989 | 0.787 | 0.029 | 0.016 |
| SVD | 0.887 | 0.695 | 0.051 | 0.028 |
| **ALS** (selected) | n/a | n/a | **0.085** | **0.084** |
| BPR | n/a | n/a | 0.080 | 0.065 |

Precision@10/Recall@10 in the 0.01–0.1 range is expected, not anomalous, for sparse top-N implicit recommendation at this scale: with 3,900 candidate movies and a per-user relevance set defined by a 20%-held-out temporal split, even a well-fit model recovers only a fraction of "relevant" items in any fixed top-10 list. The popularity baseline is *intentionally* the weakest model here — it is the non-personalized control every other model is measured against, not a bug to be fixed.

**Discussion:** SVD improves meaningfully over the popularity baseline on rating-prediction error (RMSE 0.887 vs. 0.989), confirming that personalization from collaborative filtering captures signal a global average misses. ALS and BPR, despite not being directly comparable on RMSE/MAE, both roughly 1.5–2.7× SVD's Precision@10/Recall@10 — implicit matrix factorization on confidence-weighted interactions is a better fit for the actual serving task (ranking a top-K list) than optimizing squared rating error, a well-known result in the recommender-systems literature and reproduced here at small scale. Between the two implicit models, ALS's reconstruction-error objective edges out BPR's direct pairwise-ranking objective on both Precision@10 and Recall@10 on this dataset — a reminder that a theoretically-closer-to-the-metric objective (BPR directly optimizes pairwise ranking) doesn't automatically win in practice; on this dataset/split, ALS's better-conditioned optimization landscape and larger effective factor count (60 vs. 30) outweigh BPR's objective-alignment advantage.

**Experiment tracking:** `src/models/train.py` logs, for every run: hyperparameters, all four metrics (where applicable), and the trained model as an MLflow pyfunc artifact with pinned `pip_requirements` (`scikit-surprise==1.1.5`, `implicit==0.7.2`, `pandas==2.2.2`, `numpy==1.26.4`, `scipy==1.13.1`) so the exact serving environment is reproducible from the MLflow artifact alone. Each run is tagged `model_family` (`popularity_baseline` / `svd` / `als` / `bpr`) so `evaluate.py` can programmatically retrieve "the latest run of each family" for comparison.

**Model selection and registration:** `src/models/evaluate.py` retrieves the most recent run of each of the four families, ranks them by `recall_at_10` (chosen because it's the one metric every model family logs, unlike RMSE/MAE which ALS/BPR don't produce), and registers the winner as a new version of the `movie-recommender-prod` model in the MLflow Model Registry, promoting it to the `Production` stage and archiving any previous production version. It then downloads that same model version's artifacts out of MLflow's own artifact store into `models/production_model/`, which the `dvc_commit_model` Airflow task DVC-tracks and pushes to the DVC remote — so the artifact registered in MLflow, the one the API loads, and the one DVC versions are provably the same file, not three independently-managed copies. In the current run, **ALS** wins and is what the API serves.

## 6. Deployment Strategy

The serving layer (`src/serving/`) is a FastAPI application, containerized via `Dockerfile.api` and run as the `api` service in `docker-compose.yml`.

- **Model loading:** at container startup (`lifespan` context manager), `RecommenderService.load()` pulls the current `Production`-stage model from the MLflow Model Registry (`models:/movie-recommender-prod/Production`) and parses `movies.dat` into an in-memory `movie_id → title` catalog. This means promoting a new model version in MLflow and restarting the `api` container is the entire deployment step — no code change required.
- **Endpoints:**
  - `GET /recommend/{user_id}?k=10` — returns up to `k` (1–100) recommended movies with titles; `400` on out-of-range `k`, `503` if the model isn't loaded, `500` on an unexpected prediction failure.
  - `GET /health` — readiness probe used by the container healthcheck and by Docker Compose's `depends_on: condition: service_healthy`.
  - `GET /metrics` — Prometheus exposition via `prometheus-fastapi-instrumentator`, auto-instrumenting request counts, latency histograms, and in-flight requests.
  - `GET /drift-metrics` — custom Prometheus gauges surfacing the latest offline drift-check result.
- **Two dependency sets:** `requirements.txt` (the full pipeline environment — PySpark, DVC, Evidently, pytest — used by the Airflow image and CI) is intentionally kept separate from `requirements-api.txt` (a minimal serving-only set) so the always-on production image stays small and doesn't carry offline-pipeline dependencies it never uses.
- **Prediction logging via Kafka:** every `/recommend` call publishes an event to the `recommendation-events` Kafka topic instead of writing directly to a file — this decouples the always-on serving path from the batch monitoring pipeline (the API keeps serving even if the consumer/monitoring side is down) and would let multiple API replicas publish to the same topic without needing a shared filesystem. The publish is wrapped in a try/except that logs a warning on failure rather than failing the request, so the logging side-channel can never take down serving. A bounded consumer (`streaming/kafka_consumer.py`, `auto_offset_reset="earliest"` with a short `consumer_timeout_ms` to know when it's caught up) drains the topic into `prediction_logs/predictions.jsonl` — the same file shape the drift/aggregation jobs read — on a schedule via the `drift_monitoring` Airflow DAG. `kafka-ui` (port 8081) gives a live visual view of the topic for demo purposes.

## 7. Monitoring Strategy

- **Prometheus** (`monitoring/prometheus.yml`) scrapes the API on two paths: `/metrics` (request rate, p50/p95 latency, error rate — all auto-instrumented) and `/drift-metrics` (`recommender_dataset_drift`, `recommender_drift_share`, `recommender_drift_samples`, `recommender_top_movie_share`), on independent scrape intervals (15s and 30s respectively) since drift/concentration status changes far less frequently than request-level metrics.
- **Grafana** is provisioned entirely as code (`monitoring/grafana/provisioning/`): a Prometheus datasource and a "Movie Recommender API" dashboard (request rate, error rate, p50/p95 latency, recommendation volume, drift flag, drift share, drift sample size, recommendation concentration) load automatically on container start — no manual dashboard-building during a demo or viva.
- **Drift detection** (`monitoring/drift_detection.py`) is the most substantive monitoring component: it uses Evidently's `DataDriftPreset` to statistically compare two distributions of the same two features (`movie_rating_count`, `movie_avg_rating`) — the **reference** distribution (every movie in the training catalog) against the **current** distribution (only the movies actually recommended in live traffic, drained from Kafka into `prediction_logs/predictions.jsonl`). If the model starts recommending a narrow slice of the catalog (e.g., always the same blockbusters, or drifting toward obscure low-count titles), this shows up as distribution drift on those two features. Below a floor of 30 logged recommendations, the check is skipped and reported as `insufficient_data` rather than producing a statistically meaningless result on too few samples.
- **Recommendation concentration** (`beam_jobs/aggregate_recommendations.py`) is a separate, complementary signal computed by a Beam DirectRunner batch job over the same drained prediction log: per-movie recommendation frequency and a `top_movie_share` (max single-movie count ÷ total recommendations) metric, written to `monitoring/recommendation_frequency.json` and re-exposed as the `recommender_top_movie_share` Prometheus gauge. This catches a specific failure mode — the model narrowing in on a small slice of the catalog — earlier and more directly than Evidently's more general distribution-drift check would, since it looks at concentration on the *recommended* side specifically rather than comparing two feature distributions.
- In a real verification run against live traffic (480 recommended-movie samples), the check correctly flagged drift (`dataset_drift=true`, `drift_share=0.5`) — recommendations skew toward the deployed model's preferred slice of already-popular titles, which is a genuine and expected signal, not a false positive, since a k=5–14 test workload sampled only 50 distinct users against a full catalog of movies.
- All three monitoring steps — Kafka drain (`consume_kafka_events`), Beam aggregation (`aggregate_recommendations`), and the Evidently drift check (`check_drift`) — run in sequence on the `drift_monitoring` Airflow DAG every 30 minutes, independent of the training pipeline, writing both a human-readable HTML report (`drift_reports/drift_report.html`) and machine-readable JSON status files (`monitoring/drift_status.json`, `monitoring/recommendation_frequency.json`) that the API re-exposes to Prometheus — so a human can inspect the detailed report while Grafana/alerting can act on the summary gauges without running Evidently or Beam directly.

## 8. CI/CD

`.github/workflows/ci-cd.yml` runs on every push and pull request:

1. **lint-and-test job:** installs `requirements.txt`, runs `flake8` (project-wide config in `.flake8`) across `src/`, `tests/`, `monitoring/`, `airflow/dags/`, `scripts/`, `spark_jobs/`, `streaming/`, `beam_jobs/`, then runs the full `pytest` suite — 33 tests covering the preprocessing transforms (against tiny synthetic inputs shaped like the real `.dat` files), all four model classes and shared metrics/data-utils, the FastAPI serving layer (health, recommend, validation, Kafka-publish prediction logging, drift-metrics — using a monkeypatched fake model and a mocked Kafka producer so no live MLflow server or broker is needed), the Kafka consumer's bounded drain logic (fed fake `ConsumerRecord`-like objects, no live broker), the Beam aggregation job (run for real against a DirectRunner pipeline over a tiny fixture file), and the drift-detection log-parsing/threshold logic.
2. **docker job** (depends on lint-and-test passing): fetches the raw dataset (needed because `Dockerfile.api` bakes `movies.dat` into the image for the title catalog) and builds `Dockerfile.api`. On pushes to `main` specifically, it additionally logs into GHCR and pushes the built image tagged `:latest` and `:<commit-sha>` to `ghcr.io/da25g506-dev/mlops-movie-recommender-project`, using the repository's automatically-provisioned `GITHUB_TOKEN` with `packages: write` permission — no manually-managed secret required.

This was verified green end-to-end on the actual repository: both jobs completed successfully, and the image was published to GHCR from a push to `main`.

## 9. Results & Discussion

The system runs as a fully reproducible, `docker compose up -d --build` local deployment. Every stage was verified against the live running system, not just "the code should work":

- The training DAG (`movie_recommender_pipeline`) completed all 6 tasks successfully end-to-end (`download_data → spark_preprocess → dvc_commit_processed → train_models → evaluate_and_register → dvc_commit_model`), including a real Spark job execution inside the Airflow worker container.
- 16+ DVC-tracked file objects were confirmed pushed to the configured local DVC remote, including the retrained production model export.
- MLflow shows 4 completed runs in the `movie-recommender` experiment with full params/metrics/artifacts (popularity, SVD, ALS, BPR), and one registered model (`movie-recommender-prod`) promoted to `Production`.
- The FastAPI service was confirmed serving real recommendations (`/recommend/1?k=5` returning actual MovieLens titles like *Rain Man (1988)*, *The Shawshank Redemption (1994)*), with `/health`, `/metrics`, and input validation all behaving correctly, and each call confirmed landing on the `recommendation-events` Kafka topic.
- The `drift_monitoring` DAG's 3 tasks (`consume_kafka_events → aggregate_recommendations → check_drift`) were confirmed running successfully in order, both via manual per-task execution and via a real Airflow trigger — the Kafka consumer drained real events into `prediction_logs/predictions.jsonl`, the Beam job produced `monitoring/recommendation_frequency.json`, and the drift check produced an updated `drift_status.json`.
- Prometheus shows both scrape targets (`movie-recommender-api`, `movie-recommender-drift`) as `UP`; Grafana's provisioned datasource and dashboard were confirmed queryable and rendering live data, including the new recommendation-concentration panel.
- The drift check ran successfully against real logged traffic and correctly flagged drift, with the result visible end-to-end through `/drift-metrics` → Prometheus → Grafana.
- GitHub Actions CI/CD ran green on a real push to `main`, including a successful image push to GHCR.

On modeling results specifically: both implicit-feedback models (ALS, BPR) substantially outperform SVD on Precision@10/Recall@10 despite SVD's better rating-prediction error, underscoring that optimizing for the actual deployment objective (ranking quality for a top-K list) rather than a proxy objective (squared rating error) matters — a model can win on one metric and lose on the one that reflects real user-facing behavior. Adding BPR as a fourth model, specifically to test whether a direct pairwise-ranking objective would beat ALS's reconstruction-error objective on this task, is itself a useful negative result: it didn't, on this dataset/split, which is exactly the kind of finding a tracked, multi-model comparison (rather than tuning a single model in isolation, or assuming a theoretically-motivated objective automatically wins in practice) is designed to surface. The BM25-weighting ablation (§5) is the same story at a smaller scale — a well-known technique that measurably underperformed on this specific dataset, kept out of the final models on evidence rather than convention.

## 10. Challenges Faced

- **MLflow artifact storage across containers.** Initially, `--default-artifact-root` pointed at a local filesystem path (`/mlflow/artifacts`) on the MLflow server's own volume. When the Airflow container's training task tried to log a model, `mlflow.pyfunc.log_model()` attempted to write directly to that path — which the Airflow container has no mount for — causing a `PermissionError`. The fix was architectural, not a permissions patch: switch the MLflow server to serve artifacts over its own HTTP API (`--serve-artifacts --artifacts-destination /mlflow/artifacts --default-artifact-root mlflow-artifacts:/`), so remote clients proxy all artifact I/O through the tracking server instead of needing direct filesystem access.
- **Host/container UID mismatches.** The Airflow container runs as uid 50000; the host directories it writes to (raw data, DVC cache) are owned by the host user's uid. Both the `download_data` and `dvc_commit_processed` tasks failed with `PermissionError` until the affected host directories were made group/other-writable.
- **DVC remote outside the bind-mounted project directory.** The DVC remote lives at a path outside the git repository root (to keep large data out of the repo checkout entirely), which meant it wasn't visible inside any container by default — `docker-compose.yml`'s bind mounts only cover paths under the project root. Fixed by adding an explicit second bind mount for that exact host path.
- **Stale registry state after an architecture change.** After the MLflow artifact-storage scheme changed, an experiment auto-created during an earlier failed attempt was left pointing at the old artifact-location scheme with zero completed runs, blocking a clean retry. Rather than deleting tracking data unilaterally, this was resolved by presenting the tradeoff explicitly and getting confirmation before running the delete — appropriate given MLflow experiment deletion is irreversible.
- **Avoided Beam's Python KafkaIO cross-language connector.** Beam's Python SDK doesn't implement Kafka I/O natively — `ReadFromKafka`/`WriteToKafka` are cross-language transforms that depend on a JVM-based "expansion service" running alongside the pipeline, adding a fragile extra moving part for a course project's demo environment. Instead, Kafka and Beam were given cleanly separated roles: a plain `kafka-python` consumer (`streaming/kafka_consumer.py`) drains the topic to a flat file, and the Beam DirectRunner job (`beam_jobs/aggregate_recommendations.py`) reads that file — Beam's portable batch-processing model is demonstrated without taking on a dependency that would make the pipeline harder to run reliably on a single machine.
- **`models/` directory initially unused.** The original plan called for DVC-tracking an exported model artifact, but the first implementation only ever logged models into MLflow's own artifact store, leaving `models/` an empty placeholder with just a `.gitkeep`. Caught during a post-build review, not during initial development — a reminder that "the model is trained and served correctly" and "every planned artifact actually exists on disk" are different claims and both need checking. Fixed by adding `export_production_model()` to `evaluate.py` (downloads the newly-promoted version's artifacts out of MLflow via `mlflow.artifacts.download_artifacts`) and a new `dvc_commit_model` Airflow task that DVC-tracks and pushes the result, verified by re-running the full DAG end-to-end and confirming `models/production_model.dvc` and a new file count in the DVC remote.

## 11. Future Improvements

- **Hyperparameter tuning** via MLflow's tracking to sweep `n_factors`/`regularization`/`iterations` for SVD and ALS rather than the fixed configuration used here, likely closing further ground for both on Precision/Recall@K.
- **Cold-start handling** for new users/movies with no rating history — currently the popularity baseline is the only model with a defined fallback for unseen entities; a hybrid approach (content-based features for cold items, blended with CF for warm ones) would generalize better.
- **A/B testing infrastructure** for comparing model versions on live traffic before fully promoting a new `Production` stage, rather than the current single-metric offline comparison.
- **Alerting rules** on top of the Prometheus drift gauges (e.g., Alertmanager) so drift/error-rate breaches page automatically rather than requiring someone to check the Grafana dashboard.
- **Scaling the Spark job to a real cluster** (e.g., a Spark-on-Kubernetes or EMR setup) — the current `spark_jobs/preprocess.py` is written to be cluster-portable but has only been run locally, given the dataset size doesn't require it.
