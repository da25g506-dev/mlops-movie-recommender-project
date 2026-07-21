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

### 3. Model development (4 models, MLflow)

Four genuinely different approaches, all evaluated on the same held-out split:

| Model | Library | What it does |
|---|---|---|
| Popularity baseline | pandas | Recommends whatever has the highest average rating (non-personalized) |
| SVD | scikit-surprise | Classic explicit-feedback matrix factorization — predicts a rating score |
| ALS | implicit | Implicit-feedback matrix factorization — fits confidence-weighted interaction data, not ratings directly; reconstruction-error objective |
| BPR | implicit | Implicit-feedback matrix factorization; directly optimizes pairwise ranking instead of reconstruction error |

The split is **per-user, temporal**: each user's most recent 20% of ratings (by timestamp) is held out, rather than a random split — because the real task is predicting *future* behavior from *past* behavior, not interpolating gaps in history.

Metrics: RMSE/MAE (rating-prediction error — doesn't apply to ALS/BPR, since neither ever models the 1–5 scale) and Precision@10/Recall@10 (ranking quality — applies to all four, and is what actually matters for "does the top-10 list contain things the user likes").

**On BM25 weighting:** the standard technique for turning explicit ratings into implicit "confidence" for ALS/BPR is BM25 re-weighting, which downweights globally-popular items. It was actually implemented and grid-searched (with/without, sweeping factors/regularization/iterations/learning_rate) on this exact train/test split — and it consistently *reduced* Precision@10/Recall@10 for both models. So both ALS and BPR use raw 1–5 ratings directly as confidence weights instead, and their hyperparameters were re-tuned under that scheme. This is a real ablation finding, not a shortcut — a commonly-recommended technique that measurably didn't help on this dataset's scale.

**Result:** ALS wins on Precision@10/Recall@10 (0.085 / 0.084), ahead of BPR (0.080 / 0.065), because it's optimizing something closer to the real deployment objective (rank quality) rather than a proxy (squared rating error) — and, somewhat counterintuitively, ALS's reconstruction-error objective still beats BPR's more directly ranking-aligned objective on this dataset, showing that a theoretically-closer objective doesn't automatically win in practice.

Every run logs params, metrics, and the model itself (as an MLflow pyfunc artifact with pinned dependencies) to MLflow. `src/models/evaluate.py` then pulls the latest run of each of the four "families" (tagged `model_family`), ranks by `recall_at_10` (the one metric all four produce), promotes the winner to the `Production` stage of a registered model called `movie-recommender-prod`, and exports that same version's artifacts to `models/production_model/` on disk so it can be DVC-tracked (see §5) rather than living only inside MLflow's own storage.

### 4. Orchestration (Airflow)

Two DAGs:

- **`movie_recommender_pipeline`** (daily): `download_data → spark_preprocess → dvc_commit_processed → train_models → evaluate_and_register → dvc_commit_model`. This is the full training pipeline, wired as a linear dependency chain.
- **`drift_monitoring`** (every 30 minutes): `consume_kafka_events → aggregate_recommendations → check_drift` — drains the Kafka topic, runs the Beam aggregation job, then checks drift against live traffic.

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
- `GET /drift-metrics` — custom gauges exposing the latest drift-check and recommendation-concentration results.
- Every `/recommend` call publishes an event to the `recommendation-events` Kafka topic instead of writing directly to a file — this decouples the always-on serving path from the batch monitoring pipeline (serving keeps working even if the consumer/monitoring side is down). The publish is wrapped in a try/except that only logs a warning on failure, so a broker hiccup can never fail a `/recommend` request.

Two separate dependency files: `requirements.txt` (the full pipeline environment — Spark, Beam, DVC, Evidently, pytest — used by Airflow and CI) vs. `requirements-api.txt` (minimal, serving-only, but includes `kafka-python` since the API needs to publish). This keeps the always-on production image small.

**Deployment model:** promoting a new model version in MLflow + restarting the `api` container *is* the deployment. No code change required to ship a new model.

### 7. Streaming decoupling (Kafka)

The FastAPI service and the monitoring pipeline used to be coupled through a shared file (`prediction_logs/predictions.jsonl`) written directly by every `/recommend` call. That's replaced with a Kafka topic, `recommendation-events`:

- The API publishes one JSON message per `/recommend` call to the topic (`kafka:29092` internally, exposed to the host on `localhost:9094`).
- A bounded consumer script, `streaming/kafka_consumer.py`, drains the topic on a schedule (called from the `drift_monitoring` DAG as the `consume_kafka_events` task): it reads with `auto_offset_reset="earliest"` and a short `consumer_timeout_ms` so it knows when it's caught up and exits cleanly rather than blocking forever, appending each message as a line to `prediction_logs/predictions.jsonl` — the same file shape the rest of the monitoring pipeline already expected, so downstream code didn't need to change.
- `kafka-ui` (port 8081) gives a browser-based view of the topic and its messages, useful for a live demo.

**Why bother?** It decouples the always-on serving path from the batch monitoring pipeline — the API keeps serving recommendations even if the consumer/Beam/Evidently side is down or backed up, and multiple API replicas could publish to the same topic without needing a shared filesystem, which a direct file write can't offer.

### 8. Batch aggregation (Beam)

`beam_jobs/aggregate_recommendations.py` is a small Apache Beam pipeline, run on the DirectRunner (in-process, no cluster needed):

- Reads `prediction_logs/predictions.jsonl` line by line.
- `FlatMap`s out the recommended `movie_id`s from each logged event.
- `Count.PerElement()` to get a per-movie recommendation frequency.
- Writes `monitoring/recommendation_frequency.json`, including a computed `top_movie_share` = (the most-recommended movie's count) ÷ (total recommendations made) — a "recommendation concentration" signal.

It's wired into the `drift_monitoring` DAG as the `aggregate_recommendations` task, between the Kafka drain and the Evidently drift check: `consume_kafka_events → aggregate_recommendations → check_drift`. The API re-exposes `top_movie_share` as a Prometheus gauge (`recommender_top_movie_share`), visualized in Grafana.

**Why Beam specifically, and why not just do this in pandas?** Partly to demonstrate Beam's own programming model (`ParDo`/`FlatMap`/`Count.PerElement`) on a genuinely different job shape from Spark's preprocessing ETL — Beam here is a small, portable batch aggregation rather than a large multi-table join/feature-engineering job. It's also a real, useful signal in its own right: recommendation concentration (is the model funneling everyone toward the same few movies?) is a different failure mode from Evidently's feature-distribution drift check, and catches it more directly.

**Why not Beam's `ReadFromKafka`, cutting out the separate consumer script?** Beam's Python SDK doesn't implement Kafka I/O natively — it's a cross-language transform requiring a JVM-based expansion service running alongside the pipeline, which is a fragile extra moving part for a project meant to run reliably on one machine for a demo/viva. Keeping Kafka consumption and Beam aggregation as two separate, simple steps avoids that dependency entirely.

### 9. Monitoring (Prometheus + Grafana + Evidently)

- **Prometheus** scrapes two paths on the API: `/metrics` (request-level stats, every 15s) and `/drift-metrics` (drift + concentration gauges, every 30s — deliberately slower since these change far less often).
- **Grafana** is provisioned entirely as code (datasource + dashboard, including a recommendation-concentration panel) — no manual UI clicking to set up.
- **Drift detection** (`monitoring/drift_detection.py`) uses Evidently's `DataDriftPreset` to compare two distributions on the same two features (`movie_rating_count`, `movie_avg_rating`):
  - **Reference** = the full training catalog.
  - **Current** = only the movies actually recommended in live traffic (drained from Kafka into the prediction log).

  If the model starts recommending a narrow slice of the catalog, this shows up as distributional drift. Below 30 logged predictions, the check reports `insufficient_data` rather than a statistically meaningless result. It writes an HTML report (human-readable) and a JSON status file (machine-readable, re-exposed by the API to Prometheus).
- **Recommendation concentration** (Beam's `top_movie_share`, §8) is a complementary, more direct signal for the same underlying concern — a narrowing model — computed independently of Evidently's more general distribution-drift statistics.

### 10. CI/CD (GitHub Actions)

Two jobs on every push/PR:

1. **`lint-and-test`**: install `requirements.txt`, run `flake8` (now including `streaming/` and `beam_jobs/`), run the 33-test `pytest` suite (preprocessing, all 4 models, the API with a mocked Kafka producer, the Kafka consumer's drain logic against fake records, the Beam job against a real DirectRunner pipeline over a tiny fixture, and drift detection).
2. **`docker`** (depends on job 1 passing): fetch the raw dataset (needed because the API image bakes in `movies.dat`), build `Dockerfile.api`, and — only on pushes to `main` — log into GHCR with the automatic `GITHUB_TOKEN` and push the image tagged `:latest` and `:<sha>`.

### 11. Documentation

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

5. Why can't ALS or BPR be scored with RMSE/MAE the way SVD and the popularity baseline can?
6. What is Precision@10 measuring, in plain language? What about Recall@10?
7. Why was a temporal, per-user split chosen over a random train/test split?
8. `evaluate.py` ranks models by `recall_at_10` specifically — why not RMSE, and why not some average of all the metrics?
9. ALS beat both SVD and BPR on Precision/Recall@10, even though BPR's objective (pairwise ranking) is theoretically closer to what Precision/Recall@K measure. What does that tell you about the relationship between an optimization objective and real-world deployment quality?
10. What's logged to MLflow for each run, and why does the model artifact get pinned `pip_requirements`?
11. BM25 confidence weighting is a standard technique for implicit-feedback models. Why isn't it used in this project's final ALS/BPR models?

<details><summary>Answer notes</summary>

5. Both are trained on implicit confidence weights, not on the 1–5 rating scale — neither produces a predicted rating, so there's no "error vs. true rating" to compute.
6. Precision@10: of the 10 movies recommended, what fraction did the user actually like (rating ≥ 4)? Recall@10: of all the movies the user actually liked, what fraction showed up in the top-10 list?
7. Because the real-world task is predicting future preference from past behavior, not interpolating within a user's own history — a random split would leak future information into training.
8. Recall@10 is the only metric all four model families produce (RMSE/MAE don't apply to ALS/BPR); using it keeps the comparison apples-to-apples across all of them.
9. Optimizing a proxy metric, or even an objective that's theoretically closer to the target metric, doesn't guarantee the best result in practice — ALS's better-conditioned optimization and larger factor count (60 vs. BPR's 30) outweighed BPR's objective-alignment advantage on this specific dataset/split. It's a caution against assuming a "more principled" objective automatically wins without actually measuring it.
10. Hyperparameters, metrics (RMSE/MAE/P@10/R@10 where applicable), and the model as a pyfunc artifact; pinned `pip_requirements` make the exact serving environment reproducible from the MLflow artifact alone, independent of whatever's installed on the loading machine.
11. It was actually tried — grid-searched with/without BM25 on the exact same train/test split — and it consistently *reduced* Precision@10/Recall@10 for both ALS and BPR. At this dataset's scale, BM25's popularity-downweighting removes signal that turns out to be useful rather than noise, so raw ratings are used as confidence weights instead. This is an evidence-based decision, not an oversight.
</details>

### C. Orchestration (Airflow)

12. Why are `movie_recommender_pipeline` and `drift_monitoring` separate DAGs instead of one combined DAG?
13. What does `LocalExecutor` mean, and why was `CeleryExecutor` not used here?
14. What are the 6 tasks in `movie_recommender_pipeline`, in order, and what does each one hand off to the next?
15. What are the 3 tasks in `drift_monitoring`, in order, and why in that order specifically?
16. What real problem did you hit with DAG discovery/sync during development, and how did you resolve it?

<details><summary>Answer notes</summary>

12. Different cadences (daily vs. every 30 min) and different failure domains — a drift-check failure shouldn't be coupled to or block a training run.
13. LocalExecutor runs tasks as local subprocesses on the scheduler machine using a single machine's resources — sufficient to demonstrate real scheduling/retries/dependencies without the extra complexity (and idle overhead) of a distributed Celery+Redis worker pool, which this data volume doesn't need.
14. download_data (fetch+verify raw files) → spark_preprocess (clean/join/feature-engineer → parquet) → dvc_commit_processed (version the processed data) → train_models (train+log 4 models to MLflow) → evaluate_and_register (compare runs, promote winner to Production) → dvc_commit_model (DVC-track and push the exported Production model artifact).
15. consume_kafka_events (drain the recommendation-events Kafka topic into the prediction log) → aggregate_recommendations (Beam job computes per-movie frequency/concentration from that same log) → check_drift (Evidently compares recommended-movie distribution against the training catalog). That order matters because both aggregate_recommendations and check_drift need the freshest possible data in the prediction log, so the Kafka drain has to happen first.
16. A newly added DAG file wasn't immediately triggerable even though `airflow dags list` showed it — `airflow dags reserialize` was needed to force the DAG into the ORM database before `airflow dags trigger` would work.
</details>

### D. Versioning (DVC)

17. What's the difference between what git tracks and what DVC tracks in this repo?
18. Where does the DVC remote live, and why was a local-directory remote chosen over a cloud remote (S3, GCS, etc.)?
19. What does `dvc status` reporting "Data and pipelines are up to date" actually verify?

<details><summary>Answer notes</summary>

17. Git tracks code and small `.dvc` pointer files. DVC tracks the actual large binaries (raw data, processed data, trained models) — the pointer files in git reference specific versions of those binaries.
18. A directory outside the repo on the local machine; chosen so DVC push/pull/versioning could be demonstrated as genuinely real without requiring cloud credentials for a course project.
19. That the currently checked-out `.dvc` pointer files match what's actually on disk and in the remote — i.e., nothing has drifted out of sync between what git says should exist and what actually exists.
</details>

### E. Serving (FastAPI + Docker)

20. What happens, mechanically, when you "deploy a new model version" in this system? What file(s) change?
21. Why are there two separate requirements files (`requirements.txt` vs `requirements-api.txt`), and why does `requirements-api.txt` include `kafka-python` even though it's the "minimal" set?
22. What does the `/recommend` endpoint do if `k=150`? If the model failed to load at startup? If the Kafka broker is unreachable when publishing an event?
23. What is `prediction_logs/predictions.jsonl` used for, downstream, and how does it get populated now that the API doesn't write it directly?

<details><summary>Answer notes</summary>

20. Nothing in code changes — you promote a new version to `Production` in the MLflow Model Registry, then restart (or recreate) the `api` container, which reloads whatever's currently in `Production` at startup.
21. To keep the always-on production image minimal (no PySpark/Beam/DVC/Evidently/pytest baked into the serving image) while the full pipeline environment (used by Airflow + CI) has everything training/monitoring needs. `kafka-python` is included in the minimal set specifically because the API needs to *publish* events — that's a serving-path responsibility, unlike consuming/aggregating, which are offline/batch responsibilities kept out of the API image.
22. `k=150` → `400` (out of the 1–100 valid range). Model not loaded → `503`. Kafka unreachable → the publish is wrapped in a try/except that logs a warning; the `/recommend` request still succeeds and returns a response — serving must not hard-depend on the logging side-channel.
23. It's the input to both drift detection and Beam's concentration aggregation. It's populated by `streaming/kafka_consumer.py`, a bounded consumer run periodically via the `drift_monitoring` DAG's `consume_kafka_events` task, which drains the `recommendation-events` Kafka topic and appends each event as a line to the file.
</details>

### F. Streaming decoupling (Kafka)

24. What problem does putting Kafka between the API and the monitoring pipeline actually solve? What would go wrong without it?
25. Why does `streaming/kafka_consumer.py` use a bounded drain (`consumer_timeout_ms`) instead of running as a continuous, always-listening consumer?
26. What does `kafka-ui` add that isn't otherwise visible, and why is it useful specifically for a demo?

<details><summary>Answer notes</summary>

24. It decouples the always-on serving path from the batch monitoring pipeline. Without it, `/recommend` would need to write directly to a shared log file, meaning the API's request path would be coupled to a filesystem write on every call, and multiple API replicas would need a shared filesystem to avoid clobbering each other's writes. With Kafka, the API just publishes and moves on — the monitoring side can be down, slow, or scaled independently without affecting serving.
25. The consumer is invoked periodically as an Airflow task (every 30 minutes), not as a long-running service — a bounded drain that exits once it's caught up (no new messages for a short timeout window) fits that "run, finish, exit" task model, rather than blocking Airflow's task slot indefinitely waiting for more messages that may not come for another 30 minutes.
26. It gives a live, visual view into the `recommendation-events` topic — messages arriving, offsets, partition state — without needing to run `kafka-console-consumer` or write a script; useful for actually *showing* that Kafka is doing real work during a demo, rather than just asserting it in narration.
</details>

### G. Batch aggregation (Beam)

27. What does `top_movie_share` measure, and what does a high value indicate about the deployed model?
28. Why is the Beam job a genuinely different "job shape" from the Spark preprocessing job, rather than just Beam duplicating what Spark already does?
29. Why does this project avoid Beam's `ReadFromKafka`/`WriteToKafka` transforms even though Beam is already reading from a Kafka-sourced file?

<details><summary>Answer notes</summary>

27. The share of all recommendations that went to the single most-recommended movie. A high value means the model is funneling many different users toward the same handful of movies rather than genuinely personalizing — a "recommendation concentration" problem distinct from, but related to, the more general feature-distribution drift Evidently checks for.
28. Spark's job is a multi-table join + feature-engineering ETL over the raw ratings/movies/users tables — a batch preprocessing pipeline. Beam's job is a small, portable aggregation (`FlatMap` + `Count.PerElement`) over an event log, run on DirectRunner with no cluster — closer in shape to a streaming-style aggregation than to Spark's ETL, demonstrating a different part of Beam's programming model.
29. Beam's Python SDK doesn't implement Kafka I/O natively — those are cross-language transforms requiring a JVM-based expansion service alongside the pipeline, an extra fragile moving part not worth taking on for a project meant to run reliably on a single machine. Keeping the Kafka consumer (plain `kafka-python`) and the Beam aggregation (reading a plain file) as separate steps avoids that dependency entirely.
</details>

### H. Monitoring & Drift Detection

30. What two features does the drift check compare, and on what two "populations"?
31. Why does the drift check refuse to run below 30 logged predictions?
32. In your own verification run, the drift check flagged `dataset_drift=true`. Is that a bug, or expected? Why?
33. Why does Prometheus scrape `/drift-metrics` on a slower interval (30s) than `/metrics` (15s)?
34. What would you need to add to turn "drift is flagged in Grafana" into "someone gets paged when drift is flagged"?
35. How does `recommender_top_movie_share` (from Beam) differ from `recommender_dataset_drift` (from Evidently) as a signal? Could one flag a problem the other misses?

<details><summary>Answer notes</summary>

30. `movie_rating_count` and `movie_avg_rating`, comparing the full training catalog (reference) against only the movies that were actually recommended in live traffic (current).
31. Below that floor, any statistical drift test is essentially noise — too few samples to say anything meaningful, so the system honestly reports `insufficient_data` instead of a misleading result.
32. Expected, not a bug — in the verification workload, only a small number of test users were queried, so the recommended-movie distribution over-represents the model's top picks relative to the full catalog, which is exactly the kind of concentration the drift check is designed to catch.
33. Drift status changes far less frequently than request-level metrics (it only updates once per drift-check run, every 30 minutes), so scraping it as often as live request metrics would be wasted work.
34. Alerting rules on top of the Prometheus gauges (e.g., via Alertmanager) that fire when `recommender_dataset_drift == 1`, wired to a notification channel — currently the system only surfaces drift visually in Grafana, nothing pages automatically.
35. `top_movie_share` looks specifically at concentration on the recommended side — is the model funneling everyone toward a small set of movies — regardless of whether the broader feature distribution looks "different" from training. Evidently's check compares two full distributions and could flag drift even if no single movie dominates (e.g., a broad shift toward higher-rated-count movies generally). They're complementary: a model could concentrate heavily on a handful of movies that happen to still resemble the training distribution's average feature values (high `top_movie_share`, low drift), or drift on average features without concentrating on any one movie (low `top_movie_share`, flagged drift).
</details>

### I. CI/CD

36. Why does the `docker` job depend (`needs:`) on `lint-and-test` passing first?
37. Why does the CI workflow re-download the raw dataset before building the Docker image, instead of just using what's in git?
38. Why is the image only pushed to GHCR on pushes to `main`, and not on every PR/branch push?
39. What credential is used to authenticate to GHCR, and why didn't you need to create/store a new secret?

<details><summary>Answer notes</summary>

36. No point spending time/resources building and pushing an image built from code that doesn't even pass lint/tests — fail fast, cheaper job first.
37. Raw data files are DVC-tracked, not committed to git — git only has the `.dvc` pointer files, so CI has to actually fetch the real data (`download.py`) before it can bake `movies.dat` into the API image.
38. To avoid publishing a new "latest" image for every branch/PR push, which would pollute the registry and could overwrite `latest` with unreviewed code; only merged/pushed `main` code should represent what's actually deployed.
39. The repo's automatically-provisioned `GITHUB_TOKEN`, scoped with `packages: write` permission in the workflow — no manually created/stored PAT needed.
</details>

### J. System-level / integration questions (good viva material)

40. Trace what happens end-to-end from "Airflow triggers `train_models`" to "a user calls `/recommend/5` and gets a response, and that event eventually shows up in a drift report." Name every system the data/model/event passes through.
41. If you wanted to add a fifth model family tomorrow, what would you need to change, and in how many places?
42. What's the single point of failure in this architecture, and how would you make it more resilient?
43. Why is "the whole system comes up with one `docker compose up -d --build`" considered a meaningful reproducibility property, not just a convenience?
44. If Recall@10 for the currently-Production model regressed badly after a bad training run, where would that regression first become visible, and how would you catch it before it reached users?
45. If the `drift_monitoring` DAG's Kafka-drain task started silently failing (e.g., broker unreachable), what's the very first symptom you'd notice, and where would you look?

<details><summary>Answer notes</summary>

40. train_models trains 4 models → logs to MLflow tracking server → evaluate_and_register reads back those runs, picks the Recall@10 winner, registers+promotes it in the MLflow Model Registry, and DVC-tracks the exported artifact → the `api` container, at its next startup, loads whatever's in the `Production` stage → user's `/recommend/5` call runs inference against that loaded model → response returned, and the call is published to the `recommendation-events` Kafka topic → the `drift_monitoring` DAG's `consume_kafka_events` task drains that topic into `prediction_logs/predictions.jsonl` → `aggregate_recommendations` (Beam) computes concentration stats → `check_drift` (Evidently) compares the recommended-movie distribution against the training catalog and writes the drift report/status, which the API re-exposes to Prometheus/Grafana.
41. Add a new training function (e.g. in `src/models/`), wrap it in the shared `RecommenderPyfunc` interface, add it to `train.py`'s logging loop with a new `model_family` tag, and it's automatically picked up by `evaluate.py`'s "latest run of each family" comparison — no changes needed to the registry/serving/API/Kafka/Beam code, since they're all model-agnostic once something is promoted to Production.
42. The single MLflow tracking server/registry — if it's down, no new model can be trained/registered, and if the API container restarts while MLflow is down, it can't load a model either. Making it resilient would mean running MLflow against a durable, non-local backend store and artifact store (e.g. a managed Postgres + object storage) with proper backup, rather than the local-container setup used here.
43. Because it's real, testable proof the system doesn't depend on manual, undocumented setup steps or hidden local state — a grader (or a future you, on a different machine) can clone the repo and get the identical system running, which is the actual point of "reproducibility" rather than just "runs on my machine."
44. It would first show up as a low `recall_at_10` metric in the new MLflow run itself — before it's ever promoted, since `evaluate.py` only promotes the *best* of the compared runs. If it did get promoted (e.g., all model families regressed equally), it would show up next in live Grafana panels (recommendation-quality proxies, if instrumented) or via a sharp change in the drift-metrics/concentration distribution once traffic starts hitting the new model — reinforcing why an A/B or canary step before full promotion (see Future Improvements in REPORT.md) would catch it earlier than an offline metric alone.
45. `prediction_logs/predictions.jsonl` would stop growing even though the API is clearly still serving requests (visible in `/metrics`/Grafana request-rate panels) — so the first symptom is a gap between "requests are happening" and "the prediction log/drift/concentration numbers aren't updating." You'd look at the `consume_kafka_events` task's Airflow logs first (it would show a connection error to the broker), then check `docker compose ps`/`kafka-ui` to confirm whether Kafka itself is actually down or just unreachable from that specific container.
</details>

---

## How to use this before your viva/demo

1. Read Part 1 top to bottom once, out loud if possible — that's the narration you'll give in the demo video.
2. Close this file, then attempt all 45 questions from memory. Note which ones you fumbled.
3. Go back to the actual code for the questions you fumbled — `docs/DEMO_SCRIPT.md` maps roughly to the same sections, so you can re-walk the system while re-reading the relevant code.
4. Re-attempt just the fumbled questions a day later — spaced repetition on the weak spots, not a full re-read.
