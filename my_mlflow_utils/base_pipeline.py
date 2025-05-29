import mlflow
from typing import Any, Dict
import os
from pathlib import Path # For robust path handling


class BasePipeline:
    """
    Base pipeline class for ML experiments using MLflow.
    """
    def __init__(self, experiment_name: str, run_name: str, data_loader: Any):
        """
        Initialize the pipeline with experiment and run names and a DataLoader instance.
        Also sets the MLflow tracking URI to a centralized mlruns directory.
        """
        # --- Set Centralized MLflow Tracking URI --- 
        try:
            # Assumes this file (base_pipeline.py) is in my_mlflow_utils,
            # and my_mlflow_utils is in the project root.
            my_mlflow_utils_dir = Path(__file__).resolve().parent
            project_root = my_mlflow_utils_dir.parent 
            central_mlruns_path = project_root / "mlruns"
            
            # Ensure the mlruns directory exists, MLflow usually creates it but doesn't hurt.
            # os.makedirs(central_mlruns_path, exist_ok=True) # Optional
            
            tracking_uri = central_mlruns_path.as_uri() # Generates a 'file:///...' URI
            mlflow.set_tracking_uri(tracking_uri)
            print(f"MLflow Tracking URI automatically set by BasePipeline to: {tracking_uri}")
        except Exception as e:
            print(f"Error setting MLflow tracking URI in BasePipeline: {e}. Using MLflow default.")
        # --- End MLflow Tracking URI Setup ---

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.data_loader = data_loader
        mlflow.set_experiment(self.experiment_name)

    def run(self, **params) -> mlflow.entities.Run:
        """
        Execute the pipeline: load data, train model, evaluate, and log to MLflow.
        """
        # Load data
        data = self.data_loader.load_data()

        with mlflow.start_run(run_name=self.run_name) as run:
            # Log parameters
            self.log_params(params)
            # Train the model
            model = self.train(data, **params)
            # Evaluate the model
            metrics = self.evaluate(model, data)
            # Log metrics
            self.log_metrics(metrics)
            # Log model artifacts
            self.log_model(model)
            return run

    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_model(self, model: Any):
        """
        Log the trained model to MLflow. Override if custom behavior is needed.
        """
        try:
            import mlflow.sklearn as mlf
            mlf.log_model(model, artifact_path="model")
        except ImportError:
            # Fallback to logging artifact files
            mlflow.log_artifacts(model, artifact_path="model")

    def train(self, data: Any, **params) -> Any:
        """
        Train the model. Must be implemented in a subclass.
        """
        raise NotImplementedError("BasePipeline.train must be implemented in subclass.")

    def evaluate(self, model: Any, data: Any) -> Dict[str, float]:
        """
        Evaluate the model and return metrics. Must be implemented in a subclass.
        """
        raise NotImplementedError("BasePipeline.evaluate must be implemented in subclass.") 