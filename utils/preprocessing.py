from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.logger import get_logger

logger = get_logger(__name__)

# Columns that are almost always useless (high cardinality IDs/text)
AUTO_DROP_PATTERNS = ['name', 'ticket', 'cabin', 'id', 'index', 'passengerid']


class DataPreprocessor:
    """Production-grade preprocessor for tabular CSV datasets.
    
    Handles missing values, categorical encoding, scaling,
    and auto-detection of task type (classification vs regression).
    Stores all fitted transformers for inference-time use.
    """

    def __init__(self):
        self.task_type: str = ''
        self.feature_names: List[str] = []
        self.categorical_columns: List[str] = []
        self.one_hot_columns: List[str] = ['location', 'condition'] # Columns to OHE
        self.numerical_columns: List[str] = []
        self.dropped_columns: List[str] = []
        self.label_encoder = LabelEncoder()
        self.target_classes: List[Any] = []
        self.scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.cat_encoders: Dict[str, LabelEncoder] = {}
        self.column_order: List[str] = []  # order after all transforms including dummy columns

    def _auto_drop_columns(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Drop columns that are unlikely to carry signal (high-cardinality IDs, free text)."""
        cols_to_drop = []
        for col in df.columns:
            if col == target_col:
                continue
            # Drop by name pattern
            if col.lower() in AUTO_DROP_PATTERNS:
                cols_to_drop.append(col)
                continue
            # Drop if >80% unique values (likely an ID or free-text column)
            if df[col].dtype == object and df[col].nunique() / len(df) > 0.8:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            logger.info(f"Auto-dropping high-cardinality/ID columns: {cols_to_drop}")
        
        self.dropped_columns = cols_to_drop
        return df.drop(columns=cols_to_drop)

    def _detect_task_type(self, y: pd.Series) -> str:
        """Detect whether this is classification or regression."""
        if y.dtype == object or y.nunique() <= 20:
            return 'classification'
        return 'regression'

    def fit_transform(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit all transformers on training data and return train/test splits.
        
        Args:
            df: Raw input DataFrame.
            target_col: Name of the target column.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) as numpy arrays.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

        df = df.copy()

        # 1. Feature Engineering & Type Casting
        for col in df.columns:
            if col.lower() == 'yearbuilt':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaNs with median before using it for subtraction
                df[col] = df[col].fillna(df[col].median())
                df['Age'] = 2024 - df[col]
                logger.info("Engineered feature 'Age' from 'YearBuilt'")

        # 2. Auto-drop useless columns
        df = self._auto_drop_columns(df, target_col)

        # 3. Separate target
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])

        # 4. Detect task type
        self.task_type = self._detect_task_type(y)
        logger.info(f"Detected task type: {self.task_type}")

        # 5. Identify column types
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.feature_names = list(X.columns)

        # 6. Impute missing values
        if self.numerical_columns:
            X[self.numerical_columns] = self.num_imputer.fit_transform(X[self.numerical_columns])
        if self.categorical_columns:
            X[self.categorical_columns] = self.cat_imputer.fit_transform(X[self.categorical_columns])

        # 7. One-Hot Encoding for specific columns
        ohe_cols = [c for c in self.categorical_columns if c.lower() in self.one_hot_columns]
        label_cols = [c for c in self.categorical_columns if c.lower() not in self.one_hot_columns]

        if ohe_cols:
            X = pd.get_dummies(X, columns=ohe_cols, prefix=ohe_cols)
            logger.info(f"One-hot encoded: {ohe_cols}")

        # 8. Label Encoding for remaining categories
        for col in label_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.cat_encoders[col] = le

        # 9. Scale numerical columns
        # Update numerical columns to include OHE bits if we want, or just the original ones
        # Usually it's better to scale everything after encoding if they are all numeric now
        self.column_order = list(X.columns)
        X[self.column_order] = self.scaler.fit_transform(X[self.column_order])

        # 10. Encode target for classification
        if self.task_type == 'classification':
            y = self.label_encoder.fit_transform(y.astype(str))
            self.target_classes = self.label_encoder.classes_.tolist()
        else:
            y = pd.to_numeric(y, errors='coerce').fillna(y.median()).values

        return train_test_split(X.values, y, test_size=0.2, random_state=42)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform inference data, filling any missing features with training defaults."""
        df = df.copy()

        # 1. Feature Engineering (must match fit_transform)
        for col in df.columns:
            if col.lower() == 'yearbuilt':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df['Age'] = 2024 - df[col]

        # 2. Handle missing/extra features
        # Add missing original features
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan

        # 3. Only keep original feature names before encoding
        df = df[self.feature_names].copy()

        # 4. Impute
        if self.numerical_columns:
            df[self.numerical_columns] = self.num_imputer.transform(df[self.numerical_columns])
        if self.categorical_columns:
            df[self.categorical_columns] = self.cat_imputer.transform(df[self.categorical_columns])

        # 5. One-Hot Encoding (consistent with fit)
        ohe_cols = [c for c in self.categorical_columns if c.lower() in self.one_hot_columns]
        if ohe_cols:
            df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols)
            # Add missing dummy columns that were in fit but not here
            for col in self.column_order:
                if col not in df.columns:
                    df[col] = 0

        # 6. Label Encoding
        label_cols = [c for c in self.categorical_columns if c.lower() not in self.one_hot_columns]
        for col in label_cols:
            if col in df.columns:
                le = self.cat_encoders[col]
                def safe_encode(val, encoder=le):
                    val = str(val) if val is not None and str(val) != 'nan' else encoder.classes_[0]
                    if val in encoder.classes_:
                        return encoder.transform([val])[0]
                    return encoder.transform([encoder.classes_[0]])[0]
                df[col] = df[col].apply(safe_encode)

        # 7. Final Order & Scale
        df = df[self.column_order]
        return self.scaler.transform(df.values)

    def decode_prediction(self, prediction) -> Any:
        """Convert raw model output back to human-readable label."""
        if self.task_type == 'classification':
            return self.label_encoder.inverse_transform([int(prediction)])[0]
        return float(prediction)

    def get_sample_input(self) -> Dict[str, Any]:
        """Return a sample input dict with realistic default values per feature."""
        sample = {}
        for col in self.feature_names:
            if col in self.numerical_columns:
                sample[col] = 0.0
            else:
                if col in self.cat_encoders:
                    sample[col] = self.cat_encoders[col].classes_[0]
                else:
                    sample[col] = "Default"
        return sample