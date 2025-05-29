import argparse
import sys
import os

# Add project root to Python path to allow importing my_mlflow_utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from my_mlflow_utils.data_loader import AutismDataLoader
from resnet_pipeline import ResNetPipeline # Import from the ResNet folder

def main():
    parser = argparse.ArgumentParser(description="Train a ResNet-50 model for Autism classification.")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the root of the Autism dataset (containing Autistic and Non_Autistic subfolders).")
    parser.add_argument("--experiment_name", type=str, default="Autism_Classification_ResNet", 
                        help="Name of the MLflow experiment.")
    parser.add_argument("--run_name", type=str, default="ResNet50_Run", 
                        help="Name of this specific MLflow run.")
    parser.add_argument("--model_name_tag", type=str, default="ResNet50", 
                        help="Tag for the model name/family in MLflow (e.g., ResNet50_FineTuned).")
    
    # Training hyperparameters
    parser.add_argument("--fc_epochs", type=int, default=10, help="Number of epochs to train only the FC layer.")
    parser.add_argument("--fine_tune_epochs", type=int, default=15, help="Number of epochs to fine-tune after unfreezing.")
    parser.add_argument("--initial_lr", type=float, default=0.001, help="Initial learning rate (for FC layer training).")
    parser.add_argument("--fine_tune_lr", type=float, default=0.0001, help="Learning rate for fine-tuning phase.")
    parser.add_argument("--num_unfreeze_blocks", type=int, default=1, choices=[0, 1, 2, 3], help="Number of ResNet blocks to unfreeze from top (0=FC only, 1=layer4, 2=layer4+3, 3=layer4+3+2).")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification.")

    args = parser.parse_args()

    print(f"Starting ResNet-50 pipeline with parameters:")
    print(f"  Dataset Path: {args.dataset_path}")
    print(f"  Experiment Name: {args.experiment_name}")
    print(f"  Run Name: {args.run_name}")
    print(f"  Model Name Tag: {args.model_name_tag}")
    print(f"  FC Epochs: {args.fc_epochs}")
    print(f"  Fine-tune Epochs: {args.fine_tune_epochs}")
    print(f"  Initial Learning Rate: {args.initial_lr}")
    print(f"  Fine-tune Learning Rate: {args.fine_tune_lr}")
    print(f"  Num Unfreeze Blocks: {args.num_unfreeze_blocks}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Num Classes: {args.num_classes}")

    # Initialize the data loader
    # Assuming standard splits: train, valid, test and classes: Autistic, Non_Autistic
    data_loader = AutismDataLoader(
        dataset_path=args.dataset_path,
        splits=["train", "valid", "test"], # Ensure these subdirectories exist under dataset_path
        classes=["Autistic", "Non_Autistic"] # Ensure these subdirectories exist under each split
    )

    # Initialize the ResNet pipeline
    pipeline = ResNetPipeline(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        data_loader=data_loader,
        num_classes=args.num_classes,
        model_name_tag=args.model_name_tag
    )

    # Parameters to pass to the pipeline's run method
    # These will be logged by BasePipeline and used by the train method
    run_params = {
        'fc_epochs': args.fc_epochs,
        'fine_tune_epochs': args.fine_tune_epochs,
        'learning_rate': args.initial_lr, # BasePipeline logs this as 'learning_rate'
        'fine_tune_lr': args.fine_tune_lr,
        'num_unfreeze_blocks': args.num_unfreeze_blocks,
        'weight_decay': args.weight_decay,
        'dataset_path': args.dataset_path, # Logged for traceability
        'model_name_tag': args.model_name_tag # Logged as a param, also used as a tag
    }

    print("Running ResNet pipeline...")
    try:
        pipeline.run(**run_params)
        print("ResNet pipeline run completed successfully.")
    except Exception as e:
        print(f"An error occurred during the ResNet pipeline run: {e}")
        # Optionally re-raise or handle more gracefully
        raise

if __name__ == "__main__":
    main() 