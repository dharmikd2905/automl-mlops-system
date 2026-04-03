import pytest
from fastapi.testclient import TestClient
import os
import tempfile
import pandas as pd

from api.main import app

client = TestClient(app)

def test_frontend_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "MLOps Platform" in response.text

# We can also add a placeholder for full /train test, but it requires actual file mocks etc.
