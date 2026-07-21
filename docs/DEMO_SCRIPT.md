# 10-Minute Demo Script

Record your screen (terminal + browser tabs for Airflow/MLflow/FastAPI-docs/Prometheus/Grafana already open). Narrate as you go — don't just click silently. Target ~10 minutes; timings below are a guide, not a hard cutoff.

Before recording: run `docker compose ps` once so everything is already up and healthy — don't burn demo time waiting for containers to boot.

---

## 1. Introduction (30s)

> "This is an end-to-end MLOps pipeline for a movie recommender system, built on the MovieLens ml-1m dataset — 1 million ratings, 6,040 users, 3,900 movies. The system covers the full lifecycle: automated data pipeline with Spark, Kafka, and Beam, multi-model training with experiment tracking, containerized deployment, and live monitoring with drift detection. Everything you'll see is running locally via Docker Compose."

Show `docker compose ps` — point out all 10 services (postgres, airflow-webserver, airflow-scheduler, mlflow, api, zookeeper, kafka, kafka-ui, prometheus, grafana) are `Up` and `healthy`.

## 2. Architecture walkthrough (60s)

Open **README.md**, scroll to the architecture diagram. Narrate the flow left to right:

> "Raw MovieLens files go through an Airflow DAG that downloads the data, runs a Spark preprocessing job, versions the output with DVC, trains four different models, and registers the best one in MLflow. That registered model is loaded by a FastAPI service, which publishes a Kafka event on every recommendation call and is scraped by Prometheus and visualized in Grafana. A second Airflow DAG runs every 30 minutes to drain that Kafka topic, run a Beam batch job that computes recommendation concentration, and check for drift in live traffic."

## 3. Data pipeline + Airflow (90s)

Open Airflow UI (`localhost:8080`, admin/admin). Show the DAGs list — point out `movie_recommender_pipeline` (daily) and `drift_monitoring` (every 30 min).

Click into `movie_recommender_pipeline` → Graph view. Narrate the 6 tasks:

> "download_data fetches and MD5-verifies the raw MovieLens files. spark_preprocess runs a PySpark job that cleans, dedupes, joins ratings with movies and users, and engineers features — one-hot genres, per-user and per-movie rating aggregates. dvc_commit_processed versions that processed dataset with DVC and pushes it to a remote. train_models trains four models — popularity, SVD, ALS, and BPR — and logs everything to MLflow. evaluate_and_register compares them, promotes the winner to Production, and exports its artifacts to models/production_model/, which the final dvc_commit_model task versions with DVC too — so the same artifact is tracked in both MLflow and DVC."

Click into a completed run, show all 6 tasks green. Optionally trigger a fresh run live (`Trigger DAG` button) if time allows, or explain it was just run for verification.

Terminal: show `dvc status` (via `docker exec` into the scheduler container) → "Data and pipelines are up to date" — proves DVC versioning is real, not just files sitting in git.

```bash
docker exec -it movie-recommender-mlops-airflow-scheduler-1 bash -c "cd /opt/airflow/project && dvc status && dvc remote list"
```

## 4. Model development + MLflow (90s)

Open MLflow UI (`localhost:5000`). Show the `movie-recommender` experiment with 4 runs.

> "Four genuinely different modeling approaches: a popularity baseline, SVD from scikit-surprise for classic explicit-feedback collaborative filtering, ALS from the implicit library for implicit-feedback matrix factorization, and BPR — also from implicit — which optimizes pairwise ranking directly instead of reconstruction error. Each run logs hyperparameters, metrics — RMSE, MAE, Precision@10, Recall@10 — and the serialized model as an artifact."

Open one run, show logged params/metrics/artifacts.

Click **Models** tab → `movie-recommender-prod` → show the latest version, stage = Production.

> "evaluate.py compares the latest run of each model family by Recall@10 and automatically promotes the winner. ALS won here, ahead of BPR — a good example of a theoretically closer-to-the-metric objective not automatically winning in practice. We also tried BM25 confidence weighting on both, a textbook technique, and measured it actually hurting Precision/Recall on this dataset, so both models use raw ratings as confidence weights instead — a real ablation, not just following convention."

## 5. Serving: FastAPI + Docker (90s)

Show `Dockerfile.api` briefly (or just mention it) — containerized, minimal `requirements-api.txt` separate from the full pipeline environment.

Open FastAPI docs (`localhost:8000/docs`) — show the endpoint list.

Terminal:

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/recommend/1?k=5"
```

> "The API loads whatever model is currently in the Production stage in MLflow at startup — promoting a new model version and restarting this container is the entire deployment step, no code change needed."

Mention prediction logging: every call publishes to the `recommendation-events` Kafka topic, decoupling serving from monitoring. Open **kafka-ui** (`localhost:8081`) and show the topic with messages landing in real time as you re-run the curl above.

## 6. Monitoring: Kafka, Beam, Prometheus + Grafana + drift (2 min)

Terminal:

```bash
curl "http://localhost:8000/metrics" | head -5
curl "http://localhost:8000/drift-metrics"
```

Open Prometheus (`localhost:9090`) → Status → Targets. Show both `movie-recommender-api` and `movie-recommender-drift` as `UP`.

Open Grafana (`localhost:3000`, admin/admin) → the provisioned "Movie Recommender API" dashboard. Point out panels: request rate, latency, error rate, the drift gauges, and recommendation concentration.

> "This dashboard is provisioned entirely as code — no manual setup. The drift_monitoring DAG runs three tasks every 30 minutes: consume_kafka_events drains the recommendation-events topic into a log file, aggregate_recommendations is a Beam DirectRunner batch job that computes per-movie recommendation frequency and a top_movie_share concentration signal, and check_drift uses Evidently to statistically compare the distribution of movies actually recommended in live traffic against the full training catalog. If the model starts narrowing in on a small slice of popular titles, that shows up both as the concentration signal from Beam and as distribution drift from Evidently."

Optionally show the HTML drift report (`drift_reports/drift_report.html`) for a more detailed view if you have it open in a browser tab.

## 7. CI/CD (60s)

Open the GitHub repo's **Actions** tab. Show a green workflow run.

> "Every push runs two jobs: lint-and-test — flake8 plus a 33-test pytest suite covering preprocessing, all four models, the API, the Kafka consumer, the Beam aggregation job, and drift detection — and a docker job that builds the API image and, on pushes to main, pushes it to GitHub Container Registry using the repo's built-in token, no manual secret needed."

Click into a run, expand a couple of steps (flake8, pytest, docker build+push) to show real output.

## 8. Wrap-up (30s)

> "That's the full loop: versioned data and models, tracked experiments across four model families, automated orchestration with Airflow, Spark, Kafka, and Beam, containerized serving, and live monitoring with drift and concentration detection, all tied together by CI/CD. Everything here is reproducible from a clean checkout with a single `docker compose up -d --build`."

---

### Recording checklist

- [ ] All 10 containers up and healthy before you hit record
- [ ] Browser tabs pre-opened: Airflow, MLflow, FastAPI `/docs`, kafka-ui, Prometheus targets page, Grafana dashboard, GitHub Actions
- [ ] Terminal font large enough to read on screen recording
- [ ] Have at least one MLflow run and one successful GitHub Actions run to point to (already true as of this session)
- [ ] Speak to *why*, not just *what* — the questions in `docs/STUDY_GUIDE.md` are good prep for narrating the "why"
