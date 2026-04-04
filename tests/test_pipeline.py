import pytest
import pandas as pd
import numpy as np
from utils.preprocessing import DataPreprocessor

def test_preprocessing_classification():
    """Test preprocessing for classification task"""
    # Create sample classification data
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'num_feature1': np.random.randn(n_samples),
        'num_feature2': np.random.randn(n_samples),
        'cat_feature': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Create classification target (binary)
    y = pd.Series(np.random.choice([0, 1], n_samples), name='target')
    
    df = pd.concat([X, y], axis=1)
    
    # Auto-detect (should be classification)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, 'target')
    
    assert preprocessor.task_type == 'classification'
    assert X_train.shape[0] + X_test.shape[0] == n_samples
    assert len(y_train) + len(y_test) == n_samples

def test_preprocessing_regression():
    """Test preprocessing for regression task"""
    # Create sample regression data with many unique values
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'num_feature1': np.random.randn(n_samples),
        'num_feature2': np.random.randn(n_samples),
        'cat_feature': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Create regression target (continuous with many unique values)
    y = pd.Series(
        3 * X['num_feature1'] + 2 * X['num_feature2'] + np.random.randn(n_samples) * 0.1,
        name='target'
    )
    
    df = pd.concat([X, y], axis=1)
    
    # Manually specify regression to avoid auto-detection issues
    preprocessor = DataPreprocessor(task_type='regression')
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, 'target')
    
    assert preprocessor.task_type == 'regression'
    assert X_train.shape[0] + X_test.shape[0] == n_samples
    assert len(y_train) + len(y_test) == n_samples
    
    # Check that target values are continuous (not integers from label encoding)
    assert y_train.dtype == float or y_train.dtype == np.float64
    assert len(np.unique(y_train)) > 10  # Should have many unique values

def test_preprocessing_mixed_features():
    """Test preprocessing with mixed numerical and categorical features"""
    np.random.seed(42)
    n_samples = 50
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'salary': np.random.uniform(30000, 100000, n_samples),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], n_samples),
        'city': np.random.choice(['New York', 'LA', 'Chicago', 'Houston'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, 'target')
    
    assert X_train.shape[1] > 0
    assert X_test.shape[1] > 0
    assert len(y_train) + len(y_test) == n_samples

def test_preprocessor_transform():
    """Test transform method after fitting"""
    np.random.seed(42)
    
    # Training data
    train_df = pd.DataFrame({
        'num_feature': np.random.randn(100),
        'cat_feature': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Test data
    test_df = pd.DataFrame({
        'num_feature': np.random.randn(20),
        'cat_feature': np.random.choice(['A', 'B', 'C', 'D'], 20),  # 'D' is new
        'target': np.random.choice([0, 1], 20)
    })
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(train_df, 'target')
    
    # Transform new data
    X_new = preprocessor.transform(test_df.drop(columns=['target']))
    
    assert X_new.shape[0] == 20
    assert X_new.shape[1] == X_train.shape[1]  # Same number of features
    
    # Check that unseen category 'D' was handled (encoded as zeros)
    assert not np.isnan(X_new).any()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])