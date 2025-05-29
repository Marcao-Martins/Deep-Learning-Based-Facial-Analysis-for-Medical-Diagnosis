import os
import argparse
import sys
import mlflow # Ensure mlflow is imported

# Add project root to sys.path to allow importing my_mlflow_utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from my_mlflow_utils.data_loader import AutismDataLoader
from ensemble_pipeline import EnsemblePipeline # Assuming ensemble_pipeline.py is in the same directory


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an ensemble of models on the Autism dataset.")
    parser.add_argument("--dataset", type=str, default="../AutismDataset",
                        help="Path to the Autism dataset directory (default: ../AutismDataset)")
    parser.add_argument("--experiment", type=str, default="Autism_Ensemble_Classification",
                        help="Name of the MLflow experiment (default: Autism_Ensemble_Classification)")
    parser.add_argument("--run-name", type=str, default="Ensemble_EffNetB5_MobileNetV2_InceptionV3",
                        help="Name of this specific ensemble run (default: Ensemble_EffNetB5_MobileNetV2_InceptionV3)")
    
    # General training parameters (can be overridden per model if needed in pipeline)
    parser.add_argument("--epochs", type=int, default=10,
                        help="General number of training epochs for each model (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="General learning rate for each model (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="General weight decay for optimizer for each model (default: 1e-5)")
    
    # Example of model-specific parameters (can be extended)
    parser.add_argument("--EfficientNetB5_epochs", type=int, help="Epochs for EfficientNetB5")
    parser.add_argument("--MobileNetV2_lr", type=float, help="LR for MobileNetV2")

    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Script PWD: {os.getcwd()}")
    print(f"Project Root (for my_mlflow_utils): {project_root}")

    dataset_path = os.path.abspath(args.dataset)
    print(f"Using dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset directory '{dataset_path}' does not exist! Please check the path.")
        return

    # Basic check for subdirectories (can be made more robust)
    expected_subdirs = ["train/Autistic", "train/Non_Autistic", 
                        "valid/Autistic", "valid/Non_Autistic", 
                        "test/Autistic", "test/Non_Autistic"]
    for subdir in expected_subdirs:
        if not os.path.exists(os.path.join(dataset_path, subdir)):
            print(f"WARNING: Expected subdirectory '{os.path.join(dataset_path, subdir)}' not found.")
            # Depending on strictness, you might want to exit here or let AutismDataLoader handle it.

    # 1. Create Data Loader (from my_mlflow_utils)
    # This loader fetches the raw file paths and labels
    autism_data_loader = AutismDataLoader(dataset_path=dataset_path)

    # 2. Create Ensemble Pipeline
    ensemble_pipeline = EnsemblePipeline(
        experiment_name=args.experiment,
        run_name=args.run_name,
        data_loader=autism_data_loader # Pass the instance here
    )

    # 3. Define parameters to pass to the pipeline's run method
    # These can be general or model-specific as parsed from args
    run_params = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
    }
    # Add model-specific params if provided
    if args.EfficientNetB5_epochs is not None:
        run_params['EfficientNetB5_epochs'] = args.EfficientNetB5_epochs
    if args.MobileNetV2_lr is not None:
        run_params['MobileNetV2_lr'] = args.MobileNetV2_lr
    # ... add more for other models/params as needed

    print(f"Starting ensemble pipeline run with params: {run_params}")

    # 4. Run the pipeline
    # The BasePipeline.run() method will handle:
    #   - Loading data via self.data_loader.load_data() (which is autism_data_loader.load_data())
    #   - Calling self.train(data, **run_params) (which is EnsemblePipeline.train)
    #   - Calling self.evaluate(output_of_train, data) (which is EnsemblePipeline.evaluate)
    #   - Logging params and metrics
    #   - Calling self.log_model(output_of_train) (which is EnsemblePipeline.log_model)
    try:
        ensemble_pipeline.run(**run_params)
        print(f"Ensemble training and evaluation completed. Run '{args.run_name}' logged to MLflow experiment '{args.experiment}'.")
    except Exception as e:
        print(f"An error occurred during the ensemble pipeline run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 