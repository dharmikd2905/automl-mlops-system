import io
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# ── Fast tests (no ML training) ──────────────────────

def test_root_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_schema_no_model_returns_404():
    if os.path.exists("models_store/artifacts.joblib"):
        os.remove("models_store/artifacts.joblib")
    response = client.get("/schema")
    assert response.status_code == 404

def test_predict_no_model_returns_400():
    if os.path.exists("models_store/artifacts.joblib"):
        os.remove("models_store/artifacts.joblib")
    response = client.post("/predict", json={"features": {"x": 1}})
    assert response.status_code == 400

def test_train_rejects_non_csv():
    response = client.post(
        "/train",
        files={"file": ("test.txt", b"not a csv", "text/plain")},
        data={"target_column": "target"}
    )
    assert response.status_code == 400

def test_train_rejects_missing_target():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    csv_bytes = df.to_csv(index=False).encode()
    response = client.post(
        "/train",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"target_column": "nonexistent_column"}
    )
    assert response.status_code == 500

# ── Slow tests (real training — marked separately) ───

@pytest.mark.slow
def test_train_classification(tiny_classification_csv):
    csv_bytes = tiny_classification_csv.to_csv(index=False).encode()
    response = client.post(
        "/train",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"target_column": "target"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"]["task_type"] == "classification"
    assert "model_name" in data["results"]
    assert "feature_names" in data["results"]
    assert "target" not in data["results"]["feature_names"]

@pytest.mark.slow
def test_train_regression(tiny_regression_csv):
    csv_bytes = tiny_regression_csv.to_csv(index=False).encode()
    response = client.post(
        "/train",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"target_column": "price"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"]["task_type"] == "regression"

@pytest.mark.slow  
def test_predict_after_training(tiny_classification_csv):
    # Train first
    csv_bytes = tiny_classification_csv.to_csv(index=False).encode()
    client.post(
        "/train",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"target_column": "target"}
    )
    # Then predict
    response = client.post(
        "/predict",
        json={"features": {"age": 30, "salary": 60}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "model_used" in data
