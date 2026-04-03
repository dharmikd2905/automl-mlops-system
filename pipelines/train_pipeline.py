import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from typing import Dict, Any

from utils.logger import get_logger
from utils.preprocessing import DataPreprocessor
from models.trainer import ModelTrainer

logger = get_logger(__name__)

class TrainingPipeline:
    def __init__(self, data_path: str, target_col: str):
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
        logger.info(f"Reading dataset from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # 1. Preprocess
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(df, self.target_col)
        task_type = self.preprocessor.task_type
        
        # 2. Train and Evaluate
        trainer = ModelTrainer(task_type)
        
        try:
            with mlflow.start_run():
                mlflow.log_param("task_type", task_type)
                mlflow.log_param("target_column", self.target_col)
                mlflow.log_param("dataset_rows", len(df))
                
                best_model, best_model_name, best_metrics = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
                
                self.model = best_model
                self.model_name = best_model_name
                self.metrics = best_metrics
                
                # 3. Log to MLflow
                mlflow.log_param("best_model", best_model_name)
                for metric_name, value in best_metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Log preprocessing and model
                artifacts = {
                    'model': best_model,
                    'preprocessor': self.preprocessor
                }
                joblib.dump(artifacts, "models_store/artifacts.joblib")
                
                mlflow.log_artifact("models_store/artifacts.joblib")
                logger.info("Pipeline completed successfully with MLFlow tracking.")
        except Exception as e:
            logger.warning(f"MLflow block failed, likely no server detected: {e}. Falling back to local training only.")
            best_model, best_model_name, best_metrics = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
            self.model = best_model
            self.model_name = best_model_name
            self.metrics = best_metrics
            artifacts = {
                'model': best_model,
                'preprocessor': self.preprocessor
            }
            joblib.dump(artifacts, "models_store/artifacts.joblib")
            logger.info("Pipeline completed successfully without MLflow.")

        return {
            "task_type": task_type,
            "model_name": best_model_name,
            "metrics": best_metrics
        }
            
    @staticmethod
    def load_artifacts():
        if os.path.exists("models_store/artifacts.joblib"):
            return joblib.load("models_store/artifacts.joblib")
        return None
