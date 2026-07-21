# Build Progress

Tracking the 10-stage build plan for this project.

| Stage | Description | Status |
|---|---|---|
| 1 | Scaffold repo + git init + push | ✅ done |
| 2 | Data pipeline (download + Spark preprocessing + DVC) | ✅ done |
| 3 | Model development + MLflow tracking | ✅ done |
| 4 | Airflow orchestration DAG | ✅ done |
| 5 | FastAPI serving + Dockerization | ✅ done |
| 6 | Monitoring (Prometheus + Grafana + drift detection) | ✅ done |
| 7 | Tests + CI/CD pipeline | ✅ done |
| 8 | Documentation (README + technical report) | ✅ done |
| 9 | End-to-end verification | ✅ done |
| 10 | Final polish + release tag | ✅ done |

## Stage 9 verification summary (2026-07-20)

All 7 `docker compose` services (postgres, airflow-webserver, airflow-scheduler, mlflow, api, prometheus, grafana) confirmed up and healthy. Checked directly against the live running system:

- **Training DAG** (`movie_recommender_pipeline`): latest manual run succeeded across all 5 tasks (`download_data`, `spark_preprocess`, `dvc_commit_processed`, `train_models`, `evaluate_and_register`).
- **Drift DAG** (`drift_monitoring`): multiple successful scheduled + manual runs on its 30-minute cadence.
- **DVC**: `dvc status` reports "Data and pipelines are up to date"; remote (`localremote`) reachable with tracked objects present.
- **MLflow**: `movie-recommender-prod` registered model has version 1 in the `Production` stage.
- **FastAPI**: `/health`, `/recommend/{user_id}?k=`, `/metrics`, `/drift-metrics` all verified returning correct live data (real MovieLens titles, real drift gauge values from a 480-sample check).
- **Prometheus**: both scrape targets (`movie-recommender-api`, `movie-recommender-drift`) report `up`.
- **Grafana**: provisioned Prometheus datasource and the "Movie Recommender API" dashboard both present via the Grafana API.
- **Tests**: `pytest tests/` → 24 passed. `flake8` → clean (exit 0).
- **CI/CD**: GitHub Actions run for the Stage 8 commit (`23a7608`) completed with `conclusion: success` for both `lint-and-test` and `docker` jobs, including the GHCR image push on `main`.

## Stage 10: final polish + release tag

Repo scanned clean before tagging: no `TODO`/`FIXME`/placeholder text remaining, no accidentally-tracked large files, working tree clean. Tagged `v1.0` — the code, tests, docs (README + REPORT), and CI/CD are all in the state verified in the Stage 9 summary above.

## Outstanding (not part of the 10-stage build plan)

The course PDF also requires a **10-minute demonstration video** covering pipeline execution, model training, API deployment, Docker, monitoring/logging, and CI/CD. This is a recording task, not a code/doc task, and hasn't been produced yet.

## Post-v1.0 fix: `models/` directory was unused

A user review after tagging `v1.0` caught that `models/` was still empty except for a `.gitkeep`, despite the README/REPORT already describing it as "DVC-tracked trained model artifacts." Root cause: `train.py`/`evaluate.py` only ever logged/registered models through MLflow's own artifact store (a Docker volume mounted solely inside the `mlflow` container) — nothing wrote to the local `models/` folder, so DVC had nothing there to track.

Fixed by adding `export_production_model()` to `src/models/evaluate.py` (downloads the newly-promoted Production version's artifacts via `mlflow.artifacts.download_artifacts` into `models/production_model/`) and a new `dvc_commit_model` Airflow task (`models/production_model` → `dvc add` + `dvc push`) appended to the end of `movie_recommender_pipeline`. Verified by re-running the full 6-task DAG end-to-end: `dvc_commit_model` succeeded, `models/production_model.dvc` was created and pushed to the DVC remote, `dvc status` reports up to date, and `pytest`/`flake8` still pass.

## Post-v1.0 expansion: Kafka + Beam, and a fourth model (2026-07-21)

The course PDF requires Airflow plus at least one of Spark/Kafka/Beam — Spark alone already satisfied this in v1.0. This round adds the other two Apache tools for real (not decoratively) and adds a fourth model family, based on two explicit follow-on requests: use all three Apache tools mentioned in the course PDF, and improve Precision@10/Recall@10 by trying better models.

**Model quality (Part A).** The plan's original assumption — that BM25 confidence weighting (`implicit.nearest_neighbours.bm25_weight`) would improve ALS's ranking metrics — was tested rather than assumed. A grid search over `factors`/`regularization`/`iterations`/`learning_rate`, with and without BM25, on the exact production `train_test_split_by_user(test_frac=0.2, seed=42)` split, showed BM25 consistently *reducing* recall_at_10 for both ALS and a newly added `BPRModel` (`src/models/bpr_model.py`, wrapping `implicit.bpr.BayesianPersonalizedRanking` — a pairwise-ranking objective instead of ALS's reconstruction-error objective). Both models were kept on raw ratings as confidence weights and re-tuned under that scheme instead: ALS → `factors=60, regularization=0.01, iterations=8`; BPR → `factors=30, regularization=0.1, iterations=200, learning_rate=0.005`. Retrained for real via a live `movie_recommender_pipeline` Airflow trigger (all 6 tasks green): ALS precision@10/recall@10 = 0.085/0.084 (still the Production winner), BPR = 0.080/0.065, SVD/Popularity unchanged.

**Kafka (Part B).** Added `zookeeper` + `kafka` (`confluentinc/cp-kafka:7.5.0`) + `kafka-ui` services to `docker-compose.yml`. `src/serving/app.py` now publishes a `recommendation-events` Kafka message on every `/recommend` call (lazy-initialized producer, publish wrapped in try/except so a broker hiccup never fails a request) instead of writing `prediction_logs/predictions.jsonl` directly. New `streaming/kafka_consumer.py` does a bounded drain (`auto_offset_reset="earliest"`, short `consumer_timeout_ms`) into that same file, run as the new `consume_kafka_events` Airflow task.

**Beam (Part C).** New `beam_jobs/aggregate_recommendations.py`, a DirectRunner pipeline computing per-movie recommendation frequency and a `top_movie_share` concentration signal from the drained prediction log, written to `monitoring/recommendation_frequency.json` and re-exposed as the `recommender_top_movie_share` Prometheus gauge. Wired into `drift_monitoring` as `consume_kafka_events → aggregate_recommendations → check_drift`. Deliberately avoided Beam's Python `ReadFromKafka`/`WriteToKafka` cross-language transforms (they require a JVM expansion service) in favor of keeping Kafka consumption and Beam aggregation as separate, simple steps.

**Verification.** All 10 containers (postgres, airflow-webserver, airflow-scheduler, mlflow, api, zookeeper, kafka, kafka-ui, prometheus, grafana) confirmed up and healthy. `/recommend` calls confirmed landing on the `recommendation-events` topic (via kafka-ui and the consumer's own logs). `drift_monitoring`'s 3 tasks verified both via direct in-container script execution and a real Airflow DAG trigger (3/3 succeeded), draining real events, producing `recommendation_frequency.json`, and updating `drift_status.json`. `movie_recommender_pipeline` verified via a real Airflow trigger (6/6 succeeded), reproducing the grid-search metrics. The retrained ALS model (new registry version, promoted to Production) confirmed actually served by `/recommend/1?k=5` after `docker compose restart api`, and DVC-tracked (`dvc add models/production_model && dvc push`). Test suite grew to 33 (`test_kafka_consumer.py`, `test_beam_aggregate.py`, `test_train.py`'s new BPR case, plus API test updates for the Kafka producer mock); `pytest`/`flake8` both clean.

README.md, REPORT.md, `docs/DEMO_SCRIPT.md`, and `docs/STUDY_GUIDE.md` updated to match — architecture diagrams, results tables, service/test counts, and self-test questions all reflect the real post-change system. A `v1.1` tag is proposed (not yet cut) given the scope of this change relative to `v1.0`.
