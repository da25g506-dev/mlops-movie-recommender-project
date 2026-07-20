# 🎬 Movie Recommender System — End-to-End MLOps Project

An end-to-end, production-style MLOps pipeline for a **movie recommendation system**, built on the [MovieLens ml-1m](https://grouplens.org/datasets/movielens/1m/) dataset (1M ratings, 6,040 users, 3,900 movies).

This project was built to satisfy the MLOps Course End-Term Project requirements: automated data pipelines (Airflow + Spark), data processing, multi-model development with experiment tracking (MLflow), dataset/model versioning (DVC), containerized deployment (FastAPI + Docker), monitoring (Prometheus + Grafana + drift detection), and CI/CD (GitHub Actions).

> Status: 🚧 Under active development. See [docs/PROGRESS.md](docs/PROGRESS.md) for build stage status.

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

*(To be completed in the documentation stage — see `REPORT.md` for the full technical report.)*

## System Architecture

*(Architecture diagram to be added in the documentation stage.)*

## Repository Structure

```
movie-recommender-mlops/
├── .github/workflows/       # CI/CD pipeline definitions
├── airflow/dags/            # Airflow DAG(s)
├── spark_jobs/               # PySpark data processing jobs
├── src/
│   ├── data/                 # Download & preprocessing scripts
│   ├── models/                # Training & evaluation scripts
│   └── serving/                # FastAPI application
├── monitoring/               # Prometheus, Grafana, drift detection
├── tests/                     # Unit tests
├── data/                       # DVC-tracked raw & processed data
├── models/                     # DVC-tracked trained models
├── docker-compose.yml
├── requirements.txt
├── REPORT.md                  # Technical report
└── README.md
```

## Setup & Installation

*(To be completed.)*

## Running the Pipelines & Services

*(To be completed.)*

## API Usage

*(To be completed.)*

## Docker Commands

*(To be completed.)*

## Monitoring

*(To be completed.)*

## CI/CD

*(To be completed.)*

## Dependencies

See [requirements.txt](requirements.txt).
