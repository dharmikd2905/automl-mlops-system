import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.label_encoder = LabelEncoder()
        
        self.numerical_cols = []
        self.categorical_cols = []
        self.task_type = None
        
    def detect_task_type(self, y: pd.Series) -> str:
        """Detect if the problem is classification or regression."""
        # Simple heuristic: if type is object/category or unique values < 20
        if y.dtype == 'object' or 'category' in str(y.dtype) or y.nunique() < 20:
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
        logger.info(f"Detected task type: {self.task_type}")
        return self.task_type

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        logger.info(f"Starting data preprocessing. Target: {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        # Separate X and y
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.detect_task_type(y)
        
        if self.task_type == 'classification':
            y_processed = self.label_encoder.fit_transform(y)
        else:
            y_processed = y.values
            
        # Identify columns
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numerical columns: {len(self.numerical_cols)}, Categorical columns: {len(self.categorical_cols)}")
        
        # Preprocess features
        X_processed_list = []
        
        if self.numerical_cols:
            num_data = self.num_imputer.fit_transform(X[self.numerical_cols])
            num_data = self.scaler.fit_transform(num_data)
            X_processed_list.append(num_data)
            
        if self.categorical_cols:
            # Convert to strictly string
            cat_data_raw = X[self.categorical_cols].astype(str)
            cat_data = self.cat_imputer.fit_transform(cat_data_raw)
            cat_data = self.encoder.fit_transform(cat_data)
            X_processed_list.append(cat_data)
            
        if not X_processed_list:
            raise ValueError("No viable features found for training.")
            
        X_processed = np.hstack(X_processed_list)
        
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
        
        logger.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_test, y_train, y_test

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data for prediction."""
        X_processed_list = []
        
        if self.numerical_cols:
            num_df = pd.DataFrame(index=X.index, columns=self.numerical_cols)
            for col in self.numerical_cols:
                num_df[col] = X[col] if col in X.columns else np.nan
            
            num_data = self.num_imputer.transform(num_df)
            num_data = self.scaler.transform(num_data)
            X_processed_list.append(num_data)
            
        if self.categorical_cols:
            cat_df = pd.DataFrame(index=X.index, columns=self.categorical_cols)
            for col in self.categorical_cols:
                cat_df[col] = X[col] if col in X.columns else 'Unknown'
                
            cat_data_raw = cat_df.astype(str)
            cat_data = self.cat_imputer.transform(cat_data_raw)
            cat_data = self.encoder.transform(cat_data)
            X_processed_list.append(cat_data)
            
        if not X_processed_list:
            raise ValueError("No viable features found for transformation.")
            
        return np.hstack(X_processed_list)
