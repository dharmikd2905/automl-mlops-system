# ⚡ AutoML MLOps Platform

[![CI/CD Pipeline](https://github.com/Prasham1706/automl-mlops-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Prasham1706/automl-mlops-system/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn)
![MLflow](https://img.shields.io/badge/MLflow-2.8-0194E2?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

An end-to-end AutoML system that accepts raw tabular data, auto-detects the task type, trains and compares multiple models, selects the best via automated evaluation, and exposes a production-ready prediction API — all containerized with full CI/CD.

---

## What this does

Upload a CSV → system auto-detects classification vs regression → trains 3 models in parallel → compares all metrics side-by-side → deploys the best model to a REST API. Experiments are tracked in MLflow.

**Benchmarks on sample dataset:**
- Classification accuracy: 94.7% (GradientBoostingClassifier)
- Training time: <10s on 1000-row dataset
- API prediction latency: <50ms

---

## Architecture

```
┌──────────────┐     CSV      ┌─────────────────┐     Artifacts    ┌──────────────┐
│   Browser    │ ──────────▶  │  FastAPI Server  │ ──────────────▶  │ Model Store  │
│  (Dashboard) │ ◀──────────  │  /train /predict │ ◀──────────────  │ (joblib)     │
└──────────────┘   JSON resp  └────────┬────────┘                   └──────────────┘
                                        │
                              ┌─────────▼────────┐
                              │  Training Pipeline│
                              │  ┌─────────────┐ │
                              │  │ Preprocessor│ │
                              │  └──────┬──────┘ │
                              │         ▼         │
                              │  ┌─────────────┐ │
                              │  │ ModelTrainer│ │
                              │  │ LR | RF | GB│ │
                              │  └──────┬──────┘ │
                              │         ▼         │
                              │  MLflow Tracking  │
                              └──────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ML | scikit-learn (LR, RF, GradientBoosting) |
| Tracking | MLflow |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions (Lint → Test → Docker Build) |
| Code Quality | Ruff |

---

## Quickstart

```bash
git clone https://github.com/Prasham1706/automl-mlops-system.git
cd automl-mlops-system
docker-compose up --build
```

Open:
- **Platform UI** → http://localhost:8000
- **MLflow** → http://localhost:5000

---

## API Reference

### `POST /train`
Upload a CSV and train models.

```bash
curl -X POST http://localhost:8000/train \
  -F "file=@data/sample_classification.csv" \
  -F "target_column=target"
```

Response:
```json
{
  "task_type": "classification",
  "model_name": "GradientBoostingClassifier",
  "metrics": { "accuracy": 0.947, "f1_score": 0.945 },
  "all_results": {
    "LogisticRegression": { "accuracy": 0.882, "f1_score": 0.880 },
    "RandomForestClassifier": { "accuracy": 0.931, "f1_score": 0.929 },
    "GradientBoostingClassifier": { "accuracy": 0.947, "f1_score": 0.945 }
  }
}
```

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 10, "feature2": "A"}}'
```

---

## Project Structure

```
automl-mlops-system/
├── api/
│   ├── main.py          # FastAPI routes
│   ├── schemas.py       # Pydantic models
│   └── templates/
│       └── index.html   # Dashboard UI
├── models/
│   └── trainer.py       # Multi-model training & evaluation
├── pipelines/
│   └── train_pipeline.py # Orchestration + MLflow logging
├── utils/
│   ├── preprocessing.py # Feature engineering
│   └── logger.py
├── tests/               # pytest suite
├── .github/workflows/
│   └── ci.yml          # Lint → Test → Docker Build
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Author

**Dharmik Dudhat** — B.Tech ICT, PDEU '26
