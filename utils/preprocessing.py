import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, task_type: Optional[str] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            task_type: Optional manual specification of task type ('classification' or 'regression').
                      If None, will auto-detect based on target column.
        """
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.label_encoder = LabelEncoder()
        
        self.numerical_cols = []
        self.categorical_cols = []
        self.task_type = task_type  # Allow manual override
        
    def detect_task_type(self, y: pd.Series) -> str:
        """
        Detect if the problem is classification or regression.
        
        Args:
            y: Target variable series
            
        Returns:
            String indicating 'classification' or 'regression'
        """
        # If task type was manually specified, use that
        if self.task_type is not None:
            logger.info(f"Using manually specified task type: {self.task_type}")
            return self.task_type
        
        # Otherwise auto-detect
        if not pd.api.types.is_numeric_dtype(y):
            # Non-numeric target is always classification
            self.task_type = 'classification'
        else:
            # For numeric targets, check number of unique values
            unique_values = y.nunique()
            
            # If few unique values (<= 5), it's likely classification
            # Otherwise it's regression
            if unique_values <= 5:
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'
        
        logger.info(f"Auto-detected task type: {self.task_type} (unique values: {y.nunique() if pd.api.types.is_numeric_dtype(y) else 'N/A'})")
        return self.task_type

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Starting data preprocessing. Target: {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        # Separate X and y
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.detect_task_type(y)
        
        # Process target variable
        if self.task_type == 'classification':
            y_processed = self.label_encoder.fit_transform(y)
            logger.info(f"Classification classes: {self.label_encoder.classes_}")
        else:
            y_processed = y.values
            logger.info(f"Regression target range: [{y.min():.3f}, {y.max():.3f}]")
            
        # Identify columns
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numerical columns: {len(self.numerical_cols)}, Categorical columns: {len(self.categorical_cols)}")
        
        if self.numerical_cols:
            logger.debug(f"Numerical columns: {self.numerical_cols}")
        if self.categorical_cols:
            logger.debug(f"Categorical columns: {self.categorical_cols}")
        
        # Preprocess features
        X_processed_list = []
        
        if self.numerical_cols:
            # Impute missing values
            num_data = self.num_imputer.fit_transform(X[self.numerical_cols])
            # Scale features
            num_data = self.scaler.fit_transform(num_data)
            X_processed_list.append(num_data)
            logger.info(f"Processed {len(self.numerical_cols)} numerical features")
            
        if self.categorical_cols:
            # Convert to strictly string
            cat_data_raw = X[self.categorical_cols].astype(str)
            # Impute missing values
            cat_data = self.cat_imputer.fit_transform(cat_data_raw)
            # One-hot encode
            cat_data = self.encoder.fit_transform(cat_data)
            X_processed_list.append(cat_data)
            logger.info(f"Processed {len(self.categorical_cols)} categorical features -> {cat_data.shape[1]} encoded features")
            
        if not X_processed_list:
            raise ValueError("No viable features found for training.")
            
        # Combine features
        X_processed = np.hstack(X_processed_list)
        logger.info(f"Final feature matrix shape: {X_processed.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42
        )
        
        logger.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_test, y_train, y_test

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data for prediction.
        
        Args:
            X: Input DataFrame to transform
            
        Returns:
            Transformed feature matrix
        """
        if not self.numerical_cols and not self.categorical_cols:
            raise ValueError("Preprocessor has not been fitted yet. Call fit_transform first.")
            
        X_processed_list = []
        
        if self.numerical_cols:
            # Ensure all numerical columns exist
            num_df = pd.DataFrame(index=X.index)
            for col in self.numerical_cols:
                if col in X.columns:
                    num_df[col] = X[col]
                else:
                    num_df[col] = np.nan
                    logger.warning(f"Column '{col}' not found in input data. Filling with NaN.")
            
            # Impute and scale
            num_data = self.num_imputer.transform(num_df)
            num_data = self.scaler.transform(num_data)
            X_processed_list.append(num_data)
            
        if self.categorical_cols:
            # Ensure all categorical columns exist
            cat_df = pd.DataFrame(index=X.index)
            for col in self.categorical_cols:
                if col in X.columns:
                    cat_df[col] = X[col]
                else:
                    cat_df[col] = 'Unknown'
                    logger.warning(f"Column '{col}' not found in input data. Filling with 'Unknown'.")
                
            # Convert to string, impute, and encode
            cat_data_raw = cat_df.astype(str)
            cat_data = self.cat_imputer.transform(cat_data_raw)
            cat_data = self.encoder.transform(cat_data)
            X_processed_list.append(cat_data)
            
        if not X_processed_list:
            raise ValueError("No viable features found for transformation.")
            
        return np.hstack(X_processed_list)
    
    def get_feature_names(self) -> list:
        """
        Get the names of transformed features.
        
        Returns:
            List of feature names after transformation
        """
        feature_names = []
        
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
            
        if self.categorical_cols and hasattr(self.encoder, 'get_feature_names_out'):
            cat_feature_names = self.encoder.get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_feature_names)
            
        return feature_names
    
    def inverse_transform_target(self, y_processed: np.ndarray) -> np.ndarray:
        """
        Inverse transform the target variable back to original labels/values.
        
        Args:
            y_processed: Processed target values
            
        Returns:
            Original target values
        """
        if self.task_type == 'classification':
            return self.label_encoder.inverse_transform(y_processed.astype(int))
        else:
            return y_processed