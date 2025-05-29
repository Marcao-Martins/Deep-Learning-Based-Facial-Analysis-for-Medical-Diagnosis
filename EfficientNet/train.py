import os
import argparse
import sys
import mlflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_mlflow_utils.data_loader import AutismDataLoader
from efficient_net_pipeline import EfficientNetPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet on Autism dataset")
    parser.add_argument("--dataset", type=str, default="../AutismDataset",
                        help="Path to the Autism dataset directory")
    parser.add_argument("--experiment", type=str, default="Autism_Classification",
                        help="Name of the MLflow experiment")
    parser.add_argument("--run-name", type=str, default="EfficientNet_B0",
                        help="Name of this specific run")
    parser.add_argument("--model-name", type=str, default="EfficientNet",
                        help="Name of the model for MLflow tagging")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", 
                                "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", 
                                "efficientnet_b6", "efficientnet_b7"],
                        help="EfficientNet model variant to use")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Ensure dataset path is absolute
    dataset_path = os.path.abspath(args.dataset)
    print(f"Using dataset path: {dataset_path}")
    
    # Verify dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset directory {dataset_path} does not exist!")
        return
        
    # Check for train/valid/test directories
    required_dirs = ['train', 'valid', 'test']
    missing_dirs = [d for d in required_dirs if not os.path.exists(os.path.join(dataset_path, d))]
    if missing_dirs:
        print(f"WARNING: Missing these directories: {missing_dirs}")
        
    # Verify class directories exist
    classes = ["Autistic", "Non_Autistic"]
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            if not os.path.exists(cls_path):
                print(f"WARNING: Missing class directory: {cls_path}")
            else:
                # Count images in the directory
                image_count = len([f for f in os.listdir(cls_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"Found {image_count} images in {cls_path}")
    
    # Create data loader
    data_loader = AutismDataLoader(dataset_path)
    
    # Load data explicitly to check
    data = data_loader.load_data()
    print("\nLoaded data splits:")
    for split, split_data in data.items():
        print(f"  {split}: {len(split_data['file_paths'])} images")
    
    if 'train' not in data or len(data['train']['file_paths']) == 0:
        print("ERROR: No training data found. Please check dataset organization.")
        return
    
    # Create and run pipeline
    pipeline = EfficientNetPipeline(
        experiment_name=args.experiment,
        run_name=args.run_name,
        data_loader=data_loader,
        model_name=args.model
    )
    
    # Run pipeline with specified parameters
    pipeline.run(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        model_name=args.model_name
    )
    
    print(f"Training completed. Run '{args.run_name}' logged to MLflow experiment '{args.experiment}'")


if __name__ == "__main__":
    main() 