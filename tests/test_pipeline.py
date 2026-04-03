import pytest
import pandas as pd
import numpy as np
from utils.preprocessing import DataPreprocessor

def test_preprocessing_classification():
    df = pd.DataFrame({
        'num_col': [1.0, 2.0, np.nan, 4.0, 5.0]*4,
        'cat_col': ['A', 'B', 'A', np.nan, 'B']*4,
        'target': [1, 0, 1, 0, 1]*4
    })
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, 'target')
    
    assert preprocessor.task_type == 'classification'
    assert len(preprocessor.numerical_cols) == 1
    assert len(preprocessor.categorical_cols) == 1
    assert X_train.shape[0] > 0

def test_preprocessing_regression():
    df = pd.DataFrame({
        'num_col': [1.0, 2.0, 3.0, 4.0, 5.0]*5,
        'target': [1.1, 2.2, 3.3, 4.4, 5.5]*5
    })
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, 'target')
    
    assert preprocessor.task_type == 'regression'
