# End-to-End MLOps System

A production-ready MLOps platform to automatically train, select, track, and deploy machine learning models based on user-uploaded structured CSV datasets.

## Features

- **Automated Preprocessing**: Handles missing values, categorical encoding, and feature scaling.
- **Problem Detection**: Automatically defines classification vs regression based on target variables.
- **Model Training**: Evaluates multiple machine learning models (RandomForest, GradientBoosting, LogisticRegression/LinearRegression).
- **Experiment Tracking**: Integrated with MLflow.
- **Deployment via API**: FastAPI provides endpoints to accept datasets and serve predictions.
- **Containerization**: Included Dockerfile and docker-compose configurations.
- **CI/CD setup**: Simple Github Actions workflow for test validations.

## Setup Instructions

### Local Execution (Without Docker)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start MLflow Server**:
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000
   ```

3. **Start FastAPI Service** (in another terminal):
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Execution with Docker Compose

1. Simply run:
   ```bash
   docker-compose up --build
   ```
   
This will spin up both the MLFlow tracking server (on port 5000) and the FastAPI frontend/backend (on port 8000).

## Usage

1. Open `http://localhost:8000/` in your browser.
2. Under "Train Model", upload your structured dataset (`.csv`). (Sample datasets are available in `data/`).
3. Enter the target column name (e.g. `target`) and click Train. Wait for the result object.
4. Go to the "Predict" section, insert a JSON struct of your features and click "Predict".

## Testing
Run unit tests with:
```bash
pytest tests/
```
