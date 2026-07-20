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
