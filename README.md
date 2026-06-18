# ⚡ AutoML MLOps Platform

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor=white)](https://github.com/dharmikdudhat/automl-mlops-system)
[![CI/CD Pipeline](https://github.com/dharmikdudhat/automl-mlops-system/actions/workflows/ci.yml/badge.svg)](https://github.com/dharmikdudhat/automl-mlops-system/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end, production-ready AutoML system that accepts raw CSV datasets, auto-detects task types (Classification/Regression), performs robust feature engineering, trains multiple architectures in parallel, and deploys the best-performing model to a high-concurrency REST API.

---

## 🚀 What this does

1.  **Smart Preprocessing**: Automatically handles missing values (median/mode), one-hot encoding for categorical data, and feature engineering (e.g., Age from YearBuilt).
2.  **Parallel Benchmarking**: Trains multiple models (Random Forest, Gradient Boosting, Ridge, Lasso, etc.) and compares them using RMSE, R2, Accuracy, and F1.
3.  **Inference Resilience**: Prediction API auto-fills missing features with training-time defaults (median) and provides detailed metadata on how data was handled.
4.  **Full MLOps Lifecycle**: Containerized with Docker, tracked with MLflow, and hardened with a 3-stage GitHub Actions CI/CD.

---

## 🏗️ Architecture

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
                              │  │ RF | GB | RI│ │
                              │  └──────┬──────┘ │
                              │         ▼         │
                              │  MLflow Tracking  │
                              └──────────────────┘
```

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) | REST endpoints + Modern UI serving |
| **ML** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Automated training + resilient inference |
| **Tracking** | ![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white) | Experiment logging & artifact versioning |
| **Containers** | ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) | Reproducible production deployment |
| **CI/CD** | ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white) | Automated Lint → Test → Build pipeline |
| **Linting** | ![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=python&logoColor=black) | High-performance code quality |
| **Testing** | ![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white) | Unit and integration test suite |
| **Validation** | ![Pydantic](https://img.shields.io/badge/Pydantic-E92067?logo=pydantic&logoColor=white) | Strict request/response data validation |

---

## 🛠️ Quickstart

```bash
# Clone the repository
git clone https://github.com/dharmikdudhat/automl-mlops-system.git
cd automl-mlops-system

# Launch services (API + MLflow)
docker-compose up --build
```

- **Dashboard**: [http://localhost:8000](http://localhost:8000)
- **MLflow UI**: [http://localhost:5000](http://localhost:5000)

---

## 👤 Author

**Dharmik Dudhat**
*B.Tech Information & Communication Technology*
**Pandit Deendayal Energy University (PDEU), 2026**
**CGPA: 8.8**

- [GitHub](https://github.com/dharmikdudhat)
- [LinkedIn](https://linkedin.com/in/dharmikdudhat)

---

## 📄 License

MIT License — feel free to use for your own portfolio.
