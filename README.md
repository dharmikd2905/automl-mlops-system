# End-to-End MLOps System

This project is a simple but practical implementation of an end-to-end machine learning system. It allows a user to upload a dataset, automatically train multiple models, select the best one, and use it for predictions through an API.

The goal of this project was to understand how ML systems work beyond notebooks and build something closer to a real-world workflow.

---

## What it does

* Takes a CSV dataset as input
* Detects whether the problem is classification or regression
* Cleans and preprocesses the data
* Trains multiple models and selects the best one
* Logs experiments using MLflow
* Exposes APIs for training and prediction using FastAPI
* Runs using Docker for easy setup

---

## Tech Stack

* Python
* scikit-learn
* FastAPI
* MLflow
* Docker

---

## How to run

### Using Docker (recommended)

```bash
docker-compose up --build
```

Then open:

* API → http://localhost:8000
* MLflow → http://localhost:5000

---

### Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start MLflow:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000
```

Start API:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## How to use

1. Open http://localhost:8000
2. Upload a CSV file
3. Enter the target column (for example: `price`)
4. Click "Train Model"

After training:

* Go to the predict section
* Enter input features as JSON

Example:

```json
{
  "area": 1600,
  "bedrooms": 3
}
```

---

## Example use case

You can try a simple dataset like house prices:

| area | bedrooms | price   |
| ---- | -------- | ------- |
| 1000 | 2        | 3000000 |
| 1500 | 3        | 4500000 |

---

## Project structure

```
api/            # FastAPI routes
pipelines/      # training logic
models/         # model code
utils/          # preprocessing
tests/          # test cases
```

---

## Notes

* Works best with structured tabular data
* Models are saved locally
* This is a learning + practical project, not a full production system

---

## Author

Dharmik Dudhat
