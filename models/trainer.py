from typing import Any, Dict

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from utils.logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    """Trains and evaluates multiple scikit-learn models for a given task type.
    
    Supports classification (LogisticRegression, RandomForest, GradientBoosting)
    and regression tasks. Automatically selects the best model based on
    accuracy (classification) or RMSE (regression).
    
    Args:
        task_type: Either 'classification' or 'regression'.
    """
    def __init__(self, task_type: str):
        """Initializes the ModelTrainer with the specified task type and its associated models."""
        self.task_type = task_type
        
        if self.task_type == 'classification':
            self.models = {
                'LogisticRegression': LogisticRegression(max_iter=1000),
                'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42)
            }
        elif self.task_type == 'regression':
            self.models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=1.0),
                'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
            }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates accuracy and weighted F1 score for classification tasks."""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates RMSE and R2 score for regression tasks."""
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2_score': float(r2_score(y_true, y_pred))
        }

    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Trains all models and returns metrics for each, along with the best model found.
        
        Returns:
            A dictionary containing:
                - best_model: The scikit-learn model object with best performance.
                - best_model_name: Name of the best model.
                - best_metrics: Metrics dictionary for the best model.
                - all_results: Dictionary of metrics for every trained model.
        """
        best_model = None
        best_model_name = ""
        best_metrics = {}
        all_results = {}
        best_score = -float('inf') if self.task_type == 'classification' else float('inf')
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            if self.task_type == 'classification':
                metrics = self.evaluate_classification(y_test, predictions)
                primary_metric = metrics['accuracy']
                all_results[name] = metrics
                
                logger.info(f"{name} metrics: {metrics}")
                if primary_metric > best_score:
                    best_score = primary_metric
                    best_model = model
                    best_model_name = name
                    best_metrics = metrics
            else:
                metrics = self.evaluate_regression(y_test, predictions)
                primary_metric = metrics['rmse']
                all_results[name] = metrics
                
                logger.info(f"{name} metrics: {metrics}")
                # For regression, lower RMSE is better
                if primary_metric < best_score:
                    best_score = primary_metric
                    best_model = model
                    best_model_name = name
                    best_metrics = metrics
                    
        logger.info(f"Best model selected: {best_model_name} with metrics: {best_metrics}")
        return {
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_metrics": best_metrics,
            "all_results": all_results
        }
