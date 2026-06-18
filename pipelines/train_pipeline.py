import json
import os
from datetime import datetime
from typing import Any, Dict

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from models.trainer import ModelTrainer
from utils.logger import get_logger
from utils.preprocessing import DataPreprocessor

logger = get_logger(__name__)

class TrainingPipeline:
    """Orchestrates the end-to-end machine learning pipeline from data loading to model registry.
    
    This pipeline handles preprocessing, model training, evaluation, and logging
    to both MLflow and a local history store.
    
    Args:
        data_path: Path to the CSV dataset.
        target_col: Name of the target column for prediction.
    """
    def __init__(self, data_path: str, target_col: str):
        """Initializes the training pipeline and configures MLflow tracking."""
        self.data_path = data_path
        self.target_col = target_col
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.metrics = {}
        self.model_name = ""
        
        os.makedirs("models_store", exist_ok=True)
        # Configure MLflow
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        try:
            mlflow.set_experiment("Automated_MLOps")
        except Exception as e:
            logger.warning(f"Could not connect to MLFlow: {e}. Executing without tracking if it's local.")

    def run(self) -> Dict[str, Any]:
        """Executes the full pipeline: load, preprocess, train, evaluate, and log.
        
        Returns:
            A dictionary containing the task type, best model name, metrics, 
            and results for all trained models.
        """
        logger.info(f"Reading dataset from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # 1. Preprocess
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(df, self.target_col)
        task_type = self.preprocessor.task_type
        
        # 2. Train and Evaluate
        trainer = ModelTrainer(task_type)
        all_results = {}
        
        try:
            with mlflow.start_run():
                mlflow.log_param("task_type", task_type)
                mlflow.log_param("target_column", self.target_col)
                mlflow.log_param("dataset_rows", len(df))
                
                training_results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
                
                self.model = training_results['best_model']
                self.model_name = training_results['best_model_name']
                self.metrics = training_results['best_metrics']
                all_results = training_results['all_results']
                
                # 3. Log to MLflow
                mlflow.log_param("best_model", self.model_name)
                for metric_name, value in self.metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Log preprocessing and model
                artifacts = {
                    'model': self.model,
                    'model_name': self.model_name,
                    'preprocessor': self.preprocessor
                }
                joblib.dump(artifacts, "models_store/artifacts.joblib")
                
                mlflow.log_artifact("models_store/artifacts.joblib")
                logger.info("Pipeline completed successfully with MLFlow tracking.")
        except Exception as e:
            logger.warning(f"MLflow block failed: {e}. Falling back to local training only.")
            training_results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
            self.model = training_results['best_model']
            self.model_name = training_results['best_model_name']
            self.metrics = training_results['best_metrics']
            all_results = training_results['all_results']
            
            artifacts = {
                'model': self.model,
                'model_name': self.model_name,
                'preprocessor': self.preprocessor
            }
            joblib.dump(artifacts, "models_store/artifacts.joblib")
            logger.info("Pipeline completed successfully without MLflow.")

        results = {
            "task_type": task_type,
            "model_name": self.model_name,
            "metrics": self.metrics,
            "all_results": all_results,
            "feature_names": self.preprocessor.feature_names,
            "dropped_columns": self.preprocessor.dropped_columns,
            "categorical_features": self.preprocessor.categorical_columns,
            "numerical_features": self.preprocessor.numerical_columns,
            "target_classes": getattr(self.preprocessor, 'target_classes', []),
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_to_history(results)
        return results

    def _save_to_history(self, results: Dict[str, Any]) -> None:
        """Saves the run results to a local JSON file for historical tracking."""
        history_path = "models_store/history.json"
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []
        
        history.append(results)
        # Keep only last 10 runs
        history = history[-10:]
        
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
            
    @staticmethod
    def load_artifacts() -> Any:
        """Loads the trained model and preprocessor from the artifact store."""
        if os.path.exists("models_store/artifacts.joblib"):
            return joblib.load("models_store/artifacts.joblib")
        return None
