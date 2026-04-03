from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import numpy as np
from typing import Dict, Any, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, task_type: str):
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
                'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
            }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def evaluate_classification(self, y_true, y_pred) -> Dict[str, float]:
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

    def evaluate_regression(self, y_true, y_pred) -> Dict[str, float]:
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2_score': float(r2_score(y_true, y_pred))
        }

    def train_and_evaluate(self, X_train, y_train, X_test, y_test) -> Tuple[Any, str, Dict[str, float]]:
        best_model = None
        best_model_name = ""
        best_metrics = {}
        best_score = -float('inf') if self.task_type == 'classification' else float('inf')
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            if self.task_type == 'classification':
                metrics = self.evaluate_classification(y_test, predictions)
                primary_metric = metrics['accuracy']
                
                logger.info(f"{name} metrics: {metrics}")
                if primary_metric > best_score:
                    best_score = primary_metric
                    best_model = model
                    best_model_name = name
                    best_metrics = metrics
            else:
                metrics = self.evaluate_regression(y_test, predictions)
                primary_metric = metrics['rmse']
                
                logger.info(f"{name} metrics: {metrics}")
                # For regression, lower RMSE is better
                if primary_metric < best_score:
                    best_score = primary_metric
                    best_model = model
                    best_model_name = name
                    best_metrics = metrics
                    
        logger.info(f"Best model selected: {best_model_name} with metrics: {best_metrics}")
        return best_model, best_model_name, best_metrics
