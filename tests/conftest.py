import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def tiny_classification_csv():
    """Minimal CSV that trains in <1 second."""
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 28, 33,
                38, 43, 48, 53, 58, 63, 26, 31, 36, 41],
        "salary": [50, 60, 70, 80, 90, 55, 65, 75, 52, 62,
                   72, 82, 92, 57, 67, 77, 51, 61, 71, 81],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                   0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    return df

@pytest.fixture  
def tiny_regression_csv():
    """Minimal CSV for regression tests."""
    df = pd.DataFrame({
        "area": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                 150, 250, 350, 450, 550, 650, 750, 850, 950, 1050],
        "rooms": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                  1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "price": [100, 200, 300, 400, 500, 150, 250, 350, 450, 550,
                  120, 220, 320, 420, 520, 160, 260, 360, 460, 560]
    })
    return df
