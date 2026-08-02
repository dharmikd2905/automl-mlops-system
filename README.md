# 🚀 AutoML MLOps System

<p align="center">
 
  <p align="center">
    Train, evaluate, track and serve ML models automatically using FastAPI, MLflow, Docker and GitHub Actions.
  </p>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

## 📖 Overview

AutoML MLOps System is an end-to-end machine learning platform that automates the core ML lifecycle — from raw CSV ingestion to model deployment.

Instead of manually preprocessing data, selecting algorithms, evaluating performance, and deploying models, this application runs those steps through a single pipeline: it detects whether the target column is a classification or regression problem, applies preprocessing rules (missing-value imputation, encoding, scaling), trains several candidate models in parallel, and serves the best one behind a REST API.

Supports both **Classification** and **Regression** tasks.

---

## ✨ Key Features

### 🤖 Automated Pipeline
- CSV ingestion via upload
- Target column selection
- Automatic task type detection (classification vs. regression)
- Rule-based preprocessing and feature encoding
- Multi-model training and automatic best-model selection

### 🧹 Data Preprocessing
- Missing value imputation (median for numerical, mode for categorical)
- High-cardinality / ID-like column auto-dropping
- One-Hot Encoding for low-cardinality categorical columns, label encoding otherwise
- Feature scaling (StandardScaler)
- Sample input + schema generation for inference

> Note: preprocessing is rule-based (heuristics for column type, cardinality, and a couple of hardcoded domain-specific encodings), not a fully generalized AutoML feature-engineering engine — good for structured tabular datasets, not a drop-in solution for every schema.

### 🧠 Multi-Model Benchmarking

**Classification**
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

**Regression**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### 📊 Evaluation
- Classification: Accuracy, Weighted F1
- Regression: RMSE, R²

### 📦 Model Management
- Joblib serialization of model + preprocessor
- Local training history (`models_store/history.json`)
- MLflow experiment tracking (params, metrics, artifacts)

### 🌐 REST API (FastAPI)
- `/train` — upload CSV + target column, runs the full pipeline
- `/predict` — run inference on a JSON feature vector
- `/schema` — returns expected feature names + sample input for the trained model
- `/history` — last 10 training runs

### 🐳 Docker Support
- Dockerfile + docker-compose (API + MLflow server)

### ✅ CI/CD
GitHub Actions pipeline: lint (ruff) → test (pytest) → docker build

---

## 📈 Verified Results

Benchmarked on the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (20,640 rows, regression):

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.746 | 0.576 |
| Ridge | 0.746 | 0.576 |
| Lasso | 1.145 | -0.0002 |
| **Random Forest Regressor** ⭐ | **0.505** | **0.805** |
| Gradient Boosting Regressor | 0.542 | 0.776 |

Best model auto-selected: **RandomForestRegressor (R² = 0.805)**

---

## 🏗 System Architecture

```text
                   CSV Dataset
                        │
                        ▼
              CSV / column validation
                        │
                        ▼
             Preprocessing (impute,
             encode, scale)
                        │
                        ▼
           Train candidate models
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
   Linear/Logistic  Random Forest   Gradient Boosting
        │
        ▼
 Compare on RMSE/R² or Accuracy/F1
        │
        ▼
    Select best model
        │
        ▼
     MLflow logging
        │
        ▼
 Save model + preprocessor (joblib)
        │
        ▼
    FastAPI inference endpoints
```

---

## 🛠 Technology Stack

| Layer | Technologies |
|---|---|
| Language | Python 3.10 |
| API | FastAPI |
| ML | scikit-learn |
| Experiment Tracking | MLflow |
| Serialization | Joblib |
| Data Processing | Pandas, NumPy |
| Testing | Pytest |
| Linting | Ruff |
| Deployment | Docker |
| CI/CD | GitHub Actions |

---

## 📂 Project Structure

```text
automl-mlops-system/
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── templates/
│
├── data/
│   ├── sample_classification.csv
│   └── sample_regression.csv
│
├── models/
│   └── trainer.py
│
├── pipelines/
│   └── train_pipeline.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_pipeline.py
│
├── utils/
│   ├── preprocessing.py
│   └── logger.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── ruff.toml
└── README.md
```

---

## 🚀 Getting Started

### Clone

```bash
git clone https://github.com/dharmikd2905/automl-mlops-system.git
cd automl-mlops-system
```

### Virtual Environment

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Linux / macOS:
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Application

**Locally (API only, no MLflow tracking):**
```bash
uvicorn api.main:app --reload
```
- App: http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

**Full stack (API + MLflow) via Docker:**
```bash
docker compose up --build
```
- Dashboard: http://localhost:8000
- MLflow UI: http://localhost:5000

> If you run the API standalone without an MLflow server reachable at `MLFLOW_TRACKING_URI`, training still works — it falls back to local-only logging after the connection attempt times out.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web dashboard |
| POST | `/train` | Upload CSV + target column, train models |
| POST | `/predict` | Run inference on a JSON feature vector |
| GET | `/schema` | Feature names + sample input for the current model |
| GET | `/history` | Last 10 training runs |

---

## 🧪 Testing

```bash
pytest -m "not slow"        # fast tests only
pytest                       # full suite, including real training runs
```

CI runs `pytest tests/ -v -m "not slow"` after lint, before the Docker build.

---

## 📊 MLflow

```bash
mlflow ui
```
Open http://localhost:5000 to browse experiments, parameters, metrics, and registered runs.

---

## 📸 Screenshots

### Dashboard screenshot
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6dc7358c-9e63-4bb9-bc2d-681b36ff39a2" />


### Training results screenshot
<img width="1894" height="824" alt="image" src="https://github.com/user-attachments/assets/a9bb8eab-2bc8-40b7-bf00-4fec539a914c" />

### MLflow dashboard screenshot
<img width="1455" height="560" alt="image" src="https://github.com/user-attachments/assets/66feb534-8b67-41bd-99f7-0799cf5bbf12" />

### Prediction results screenshot
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b0934f9d-11c0-455f-bca0-b92c24b87841" />


---

## 🔒 Highlights

- Modular codebase (api / pipelines / models / utils separated)
- Rule-based automatic preprocessing
- Multi-model benchmarking with automatic best-model selection
- MLflow experiment tracking
- Dockerized deployment
- CI/CD pipeline (lint → test → build)
- Unit + integration test suite

---

## 🗺 Roadmap

- Hyperparameter optimization
- XGBoost / LightGBM support
- Model versioning
- User authentication
- Dataset versioning
- Explainable AI (SHAP/LIME)
- Drift detection
- Monitoring dashboard

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/new-feature`
3. Commit: `git commit -m "Add new feature"`
4. Push: `git push origin feature/new-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see the `LICENSE` file for details.

---

## 👨‍💻 Author

**Dharmik Dudhat**
Computer Engineering Undergraduate · Full Stack & MLOps

GitHub: [github.com/dharmikd2905](https://github.com/dharmikd2905)
