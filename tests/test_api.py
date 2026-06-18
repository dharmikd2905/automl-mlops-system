import io

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_root_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_train_with_valid_csv():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
        "feature2": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1]
    })
    csv_bytes = df.to_csv(index=False).encode()
    response = client.post(
        "/train",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"target_column": "target"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data["results"]
    assert "all_results" in data["results"]

def test_train_rejects_non_csv():
    response = client.post(
        "/train",
        files={"file": ("test.txt", b"not a csv", "text/plain")},
        data={"target_column": "target"}
    )
    assert response.status_code == 400

def test_predict_without_trained_model_returns_400():
    import os
    # ensure no model exists
    if os.path.exists("models_store/artifacts.joblib"):
        os.remove("models_store/artifacts.joblib")
    response = client.post("/predict", json={"features": {"x": 1}})
    assert response.status_code == 400
