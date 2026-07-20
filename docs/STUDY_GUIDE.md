# Study Guide: What Was Built, and How to Test Your Understanding

This document has two parts: (1) a plain-language walkthrough of everything in the system and why it's built that way, and (2) a self-test questionnaire organized by topic. Try to answer each question from memory/understanding before checking the code or the answer notes at the bottom of each section — that's the point of a viva-prep exercise.

---

## Part 1: What Was Built and Why

### 1. The problem and the data

You're building a system that, given a user ID, recommends movies they're likely to enjoy. The dataset is MovieLens ml-1m: 1 million ratings from 6,040 users on 3,900 movies, stored as three `::`-delimited files (`ratings.dat`, `movies.dat`, `users.dat`).

`src/data/download.py` fetches this from GroupLens and **verifies an MD5 checksum** before trusting the download — a small but real piece of data-pipeline hygiene: fail fast on a corrupted download rather than silently training on garbage.

### 2. Data processing (Spark)

`spark_jobs/preprocess.py` is a PySpark job that:
- Cleans each of the three tables (drop nulls, drop out-of-range ratings, dedupe).
- Joins ratings → movies → users into one wide table.
- Engineers features: 18 one-hot genre columns, per-movie aggregates (rating count, average rating), per-user aggregates (rating count, average rating), and a rating year extracted from the timestamp.

**Why Spark, when the data fits in memory?** Because the *shape* of the work — multi-table joins, groupby aggregates, one-hot encoding over a ratings log — is exactly the shape of batch feature engineering at real scale (imagine millions of users instead of 6,040). Spark demonstrates that this job would scale to a cluster with no code changes beyond the master URL. A pandas mirror (`src/data/preprocess_pandas.py`) exists purely so unit tests don't need a JVM.

### 3. Model development (3 models, MLflow)

Three genuinely different approaches, all evaluated on the same held-out split:

| Model | Library | What it does |
|---|---|---|
| Popularity baseline | pandas | Recommends whatever has the highest average rating (non-personalized) |
| SVD | scikit-surprise | Classic explicit-feedback matrix factorization — predicts a rating score |
| ALS | implicit | Implicit-feedback matrix factorization — fits confidence-weighted interaction data, not ratings directly |

The split is **per-user, temporal**: each user's most recent 20% of ratings (by timestamp) is held out, rather than a random split — because the real task is predicting *future* behavior from *past* behavior, not interpolating gaps in history.

Metrics: RMSE/MAE (rating-prediction error — doesn't apply to ALS, since it never models the 1–5 scale) and Precision@10/Recall@10 (ranking quality — applies to all three, and is what actually matters for "does the top-10 list contain things the user likes").

**Result:** ALS wins on Precision@10/Recall@10 despite having no RMSE/MAE, because it's optimizing something closer to the real deployment objective (rank quality) rather than a proxy (squared rating error).

Every run logs params, metrics, and the model itself (as an MLflow pyfunc artifact with pinned dependencies) to MLflow. `src/models/evaluate.py` then pulls the latest run of each of the three "families" (tagged `model_family`), ranks by `recall_at_10` (the one metric all three produce), promotes the winner to the `Production` stage of a registered model called `movie-recommender-prod`, and exports that same version's artifacts to `models/production_model/` on disk so it can be DVC-tracked (see §5) rather than living only inside MLflow's own storage.

### 4. Orchestration (Airflow)

Two DAGs:

- **`movie_recommender_pipeline`** (daily): `download_data → spark_preprocess → dvc_commit_processed → train_models → evaluate_and_register`. This is the full training pipeline, wired as a linear dependency chain.
- **`drift_monitoring`** (every 30 minutes): a single task that runs the drift-detection script against live traffic.

They're separate DAGs, not one, because they have different cadences and different failure domains — a drift-check hiccup shouldn't block or depend on a training run.

Airflow runs with `LocalExecutor` and a Postgres metadata DB — enough to show real scheduling, retries, and dependency management, without the operational weight of `CeleryExecutor`+Redis that this data volume doesn't need.

### 5. Versioning (DVC)

DVC tracks the raw `.dat` files, the processed parquet/csv, and trained model artifacts — the large binary files that shouldn't live in git. Git only stores small `.dvc` pointer files. `scripts/dvc_commit.py` (an Airflow task) runs `dvc add` + `dvc push` to a **local-directory remote** — no cloud credentials needed, but it's a real remote with real push/pull semantics.

### 6. Serving (FastAPI + Docker)

`src/serving/app.py`, containerized via `Dockerfile.api`:

- On startup, loads whichever model is currently in the `Production` stage from the MLflow Model Registry, plus the movie catalog from `movies.dat`.
- `GET /recommend/{user_id}?k=` — top-K recommendations (k validated 1–100, else `400`).
- `GET /health` — used by the container healthcheck.
- `GET /metrics` — Prometheus exposition (auto-instrumented request rate/latency/errors).
- `GET /drift-metrics` — custom gauges exposing the latest drift-check result.
- Every `/recommend` call is appended to `prediction_logs/predictions.jsonl` — this log is both an audit trail and the input to drift detection.

Two separate dependency files: `requirements.txt` (the full pipeline environment — Spark, DVC, Evidently, pytest — used by Airflow and CI) vs. `requirements-api.txt` (minimal, serving-only). This keeps the always-on production image small.

**Deployment model:** promoting a new model version in MLflow + restarting the `api` container *is* the deployment. No code change required to ship a new model.

### 7. Monitoring (Prometheus + Grafana + Evidently)

- **Prometheus** scrapes two paths on the API: `/metrics` (request-level stats, every 15s) and `/drift-metrics` (drift gauges, every 30s — deliberately slower since drift status changes far less often).
- **Grafana** is provisioned entirely as code (datasource + a 7-panel dashboard) — no manual UI clicking to set up.
- **Drift detection** (`monitoring/drift_detection.py`) uses Evidently's `DataDriftPreset` to compare two distributions on the same two features (`movie_rating_count`, `movie_avg_rating`):
  - **Reference** = the full training catalog.
  - **Current** = only the movies actually recommended in live traffic (pulled from the prediction log).

  If the model starts recommending a narrow slice of the catalog, this shows up as distributional drift. Below 30 logged predictions, the check reports `insufficient_data` rather than a statistically meaningless result. It writes an HTML report (human-readable) and a JSON status file (machine-readable, re-exposed by the API to Prometheus).

### 8. CI/CD (GitHub Actions)

Two jobs on every push/PR:

1. **`lint-and-test`**: install `requirements.txt`, run `flake8`, run the 24-test `pytest` suite.
2. **`docker`** (depends on job 1 passing): fetch the raw dataset (needed because the API image bakes in `movies.dat`), build `Dockerfile.api`, and — only on pushes to `main` — log into GHCR with the automatic `GITHUB_TOKEN` and push the image tagged `:latest` and `:<sha>`.

### 9. Documentation

`README.md` (setup/usage-focused) and `REPORT.md` (the deeper technical report — problem statement, dataset, architecture, pipeline, models/results, deployment, monitoring, CI/CD, discussion, challenges, future work).

---

## Part 2: Self-Test Questionnaire

Try answering from memory first. Short answer notes are given after each section — cover them until you've attempted the question.

### A. Data & Preprocessing

1. Why does `download.py` verify an MD5 checksum before proceeding? What would happen if it didn't and the download was silently truncated?
2. Why is the pandas preprocessing mirror used in unit tests instead of the actual Spark job?
3. Name three feature-engineering steps performed in `spark_jobs/preprocess.py`. Why is per-movie rating count a useful feature at all — what would you use it for?
4. If ml-1m fits comfortably in memory, what's the actual justification for using Spark here?

<details><summary>Answer notes</summary>

1. To fail fast on a corrupted/partial download rather than silently training on incomplete or garbage data — a data-quality gate at the pipeline's entry point.
2. So tests run fast and don't need a JVM/PySpark installed in the test environment; it mirrors the same transformation logic on the same tiny synthetic fixtures.
3. One-hot genre encoding, per-user/per-movie rating count & average, rating year extraction. Rating count matters because it separates "genuinely well-liked" from "one lucky 5-star rating" — used later as one of the two features drift detection watches.
4. The transformations (joins, groupby aggregates, one-hot encoding) are the same shape as batch ETL on real-world-scale rating logs; Spark demonstrates the distributed-ETL pattern the pipeline would need at real scale, even though this dataset itself doesn't require it.
</details>

### B. Models & Evaluation

5. Why can't ALS be scored with RMSE/MAE the way SVD and the popularity baseline can?
6. What is Precision@10 measuring, in plain language? What about Recall@10?
7. Why was a temporal, per-user split chosen over a random train/test split?
8. `evaluate.py` ranks models by `recall_at_10` specifically — why not RMSE, and why not some average of all four metrics?
9. ALS beat SVD on Precision/Recall@10 despite SVD being tuned to minimize rating error. What does that tell you about the relationship between an optimization objective and real-world deployment quality?
10. What's logged to MLflow for each run, and why does the model artifact get pinned `pip_requirements`?

<details><summary>Answer notes</summary>

5. ALS is trained on implicit confidence weights, not on the 1–5 rating scale — it never produces a predicted rating, so there's no "error vs. true rating" to compute.
6. Precision@10: of the 10 movies recommended, what fraction did the user actually like (rating ≥ 4)? Recall@10: of all the movies the user actually liked, what fraction showed up in the top-10 list?
7. Because the real-world task is predicting future preference from past behavior, not interpolating within a user's own history — a random split would leak future information into training.
8. Recall@10 is the only metric all three model families produce (RMSE/MAE don't apply to ALS); using it keeps the comparison apples-to-apples across all three.
9. Optimizing a proxy metric (rating-prediction error) doesn't guarantee the best ranking quality, which is what users actually experience (a top-10 list). It's a caution against over-trusting one metric when the deployment objective is different.
10. Hyperparameters, metrics (RMSE/MAE/P@10/R@10 where applicable), and the model as a pyfunc artifact; pinned `pip_requirements` make the exact serving environment reproducible from the MLflow artifact alone, independent of whatever's installed on the loading machine.
</details>

### C. Orchestration (Airflow)

11. Why are `movie_recommender_pipeline` and `drift_monitoring` separate DAGs instead of one combined DAG?
12. What does `LocalExecutor` mean, and why was `CeleryExecutor` not used here?
13. What are the 5 tasks in `movie_recommender_pipeline`, in order, and what does each one hand off to the next?
14. What real problem did you hit with DAG discovery/sync during development, and how did you resolve it?

<details><summary>Answer notes</summary>

11. Different cadences (daily vs. every 30 min) and different failure domains — a drift-check failure shouldn't be coupled to or block a training run.
12. LocalExecutor runs tasks as local subprocesses on the scheduler machine using a single machine's resources — sufficient to demonstrate real scheduling/retries/dependencies without the extra complexity (and idle overhead) of a distributed Celery+Redis worker pool, which this data volume doesn't need.
13. download_data (fetch+verify raw files) → spark_preprocess (clean/join/feature-engineer → parquet) → dvc_commit_processed (version the processed data) → train_models (train+log 3 models to MLflow) → evaluate_and_register (compare runs, promote winner to Production).
14. A newly added DAG file wasn't immediately triggerable even though `airflow dags list` showed it — `airflow dags reserialize` was needed to force the DAG into the ORM database before `airflow dags trigger` would work.
</details>

### D. Versioning (DVC)

15. What's the difference between what git tracks and what DVC tracks in this repo?
16. Where does the DVC remote live, and why was a local-directory remote chosen over a cloud remote (S3, GCS, etc.)?
17. What does `dvc status` reporting "Data and pipelines are up to date" actually verify?

<details><summary>Answer notes</summary>

15. Git tracks code and small `.dvc` pointer files. DVC tracks the actual large binaries (raw data, processed data, trained models) — the pointer files in git reference specific versions of those binaries.
16. A directory outside the repo on the local machine; chosen so DVC push/pull/versioning could be demonstrated as genuinely real without requiring cloud credentials for a course project.
17. That the currently checked-out `.dvc` pointer files match what's actually on disk and in the remote — i.e., nothing has drifted out of sync between what git says should exist and what actually exists.
</details>

### E. Serving (FastAPI + Docker)

18. What happens, mechanically, when you "deploy a new model version" in this system? What file(s) change?
19. Why are there two separate requirements files (`requirements.txt` vs `requirements-api.txt`)?
20. What does the `/recommend` endpoint do if `k=150`? If the model failed to load at startup?
21. What is `prediction_logs/predictions.jsonl` used for, downstream?

<details><summary>Answer notes</summary>

18. Nothing in code changes — you promote a new version to `Production` in the MLflow Model Registry, then restart (or recreate) the `api` container, which reloads whatever's currently in `Production` at startup.
19. To keep the always-on production image minimal (no PySpark/DVC/Evidently/pytest baked into the serving image) while the full pipeline environment (used by Airflow + CI) has everything training/monitoring needs.
20. `k=150` → `400` (out of the 1–100 valid range). Model not loaded → `503`.
21. It's the input to drift detection — Evidently compares the distribution of movies that actually appear in this log against the training reference catalog.
</details>

### F. Monitoring & Drift Detection

22. What two features does the drift check compare, and on what two "populations"?
23. Why does the drift check refuse to run below 30 logged predictions?
24. In your own verification run, the drift check flagged `dataset_drift=true`. Is that a bug, or expected? Why?
25. Why does Prometheus scrape `/drift-metrics` on a slower interval (30s) than `/metrics` (15s)?
26. What would you need to add to turn "drift is flagged in Grafana" into "someone gets paged when drift is flagged"?

<details><summary>Answer notes</summary>

22. `movie_rating_count` and `movie_avg_rating`, comparing the full training catalog (reference) against only the movies that were actually recommended in live traffic (current).
23. Below that floor, any statistical drift test is essentially noise — too few samples to say anything meaningful, so the system honestly reports `insufficient_data` instead of a misleading result.
24. Expected, not a bug — in the verification workload, only a small number of test users were queried, so the recommended-movie distribution over-represents the model's top picks relative to the full catalog, which is exactly the kind of concentration the drift check is designed to catch.
25. Drift status changes far less frequently than request-level metrics (it only updates once per drift-check run, every 30 minutes), so scraping it as often as live request metrics would be wasted work.
26. Alerting rules on top of the Prometheus gauges (e.g., via Alertmanager) that fire when `recommender_dataset_drift == 1`, wired to a notification channel — currently the system only surfaces drift visually in Grafana, nothing pages automatically.
</details>

### G. CI/CD

27. Why does the `docker` job depend (`needs:`) on `lint-and-test` passing first?
28. Why does the CI workflow re-download the raw dataset before building the Docker image, instead of just using what's in git?
29. Why is the image only pushed to GHCR on pushes to `main`, and not on every PR/branch push?
30. What credential is used to authenticate to GHCR, and why didn't you need to create/store a new secret?

<details><summary>Answer notes</summary>

27. No point spending time/resources building and pushing an image built from code that doesn't even pass lint/tests — fail fast, cheaper job first.
28. Raw data files are DVC-tracked, not committed to git — git only has the `.dvc` pointer files, so CI has to actually fetch the real data (`download.py`) before it can bake `movies.dat` into the API image.
29. To avoid publishing a new "latest" image for every branch/PR push, which would pollute the registry and could overwrite `latest` with unreviewed code; only merged/pushed `main` code should represent what's actually deployed.
30. The repo's automatically-provisioned `GITHUB_TOKEN`, scoped with `packages: write` permission in the workflow — no manually created/stored PAT needed.
</details>

### H. System-level / integration questions (good viva material)

31. Trace what happens end-to-end from "Airflow triggers `train_models`" to "a user calls `/recommend/5` and gets a response." Name every system the data/model passes through.
32. If you wanted to add a fourth model family tomorrow, what would you need to change, and in how many places?
33. What's the single point of failure in this architecture, and how would you make it more resilient?
34. Why is "the whole system comes up with one `docker compose up -d --build`" considered a meaningful reproducibility property, not just a convenience?
35. If Recall@10 for the currently-Production model regressed badly after a bad training run, where would that regression first become visible, and how would you catch it before it reached users?

<details><summary>Answer notes</summary>

31. train_models trains 3 models → logs to MLflow tracking server → evaluate_and_register reads back those runs, picks the Recall@10 winner, registers+promotes it in the MLflow Model Registry → the `api` container, at its next startup, loads whatever's in the `Production` stage → user's `/recommend/5` call runs inference against that loaded model → response returned, and the call is appended to `prediction_logs/predictions.jsonl`.
32. Add a new training function (e.g. in `src/models/`), wrap it in the shared `RecommenderPyfunc` interface, add it to `train.py`'s logging loop with a new `model_family` tag, and it's automatically picked up by `evaluate.py`'s "latest run of each family" comparison — no changes needed to the registry/serving/API code, since they're all model-agnostic once something is promoted to Production.
33. The single MLflow tracking server/registry — if it's down, no new model can be trained/registered, and if the API container restarts while MLflow is down, it can't load a model either. Making it resilient would mean running MLflow against a durable, non-local backend store and artifact store (e.g. a managed Postgres + object storage) with proper backup, rather than the local-container setup used here.
34. Because it's real, testable proof the system doesn't depend on manual, undocumented setup steps or hidden local state — a grader (or a future you, on a different machine) can clone the repo and get the identical system running, which is the actual point of "reproducibility" rather than just "runs on my machine."
35. It would first show up as a low `recall_at_10` metric in the new MLflow run itself — before it's ever promoted, since `evaluate.py` only promotes the *best* of the compared runs. If it did get promoted (e.g., all three families regressed equally), it would show up next in live Grafana panels (recommendation-quality proxies, if instrumented) or via a sharp change in the drift-metrics distribution once traffic starts hitting the new model — reinforcing why an A/B or canary step before full promotion (see Future Improvements in REPORT.md) would catch it earlier than an offline metric alone.
</details>

---

## How to use this before your viva/demo

1. Read Part 1 top to bottom once, out loud if possible — that's the narration you'll give in the demo video.
2. Close this file, then attempt all 35 questions from memory. Note which ones you fumbled.
3. Go back to the actual code for the questions you fumbled — `docs/DEMO_SCRIPT.md` maps roughly to the same 8 sections, so you can re-walk the system while re-reading the relevant code.
4. Re-attempt just the fumbled questions a day later — spaced repetition on the weak spots, not a full re-read.
