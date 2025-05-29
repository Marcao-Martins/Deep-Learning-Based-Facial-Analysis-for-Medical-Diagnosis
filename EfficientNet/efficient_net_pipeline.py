import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
from tqdm import tqdm
import mlflow
from typing import Dict, Any, List, Tuple

from my_mlflow_utils import BasePipeline
from my_mlflow_utils.data_loader import AutismDataLoader
from my_mlflow_utils.mlflow_utils import log_standard_classification_metrics


class AutismImageDataset(Dataset):
    """PyTorch Dataset for loading Autism dataset images"""
    
    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Image file not found {img_path}, returning placeholder.")
            # Return a placeholder tensor and label that matches expected types/shapes
            # Adjust size if your model expects something different consistently for placeholders
            return torch.zeros((3, self.transform.transforms[0].size[0] if hasattr(self.transform.transforms[0], 'size') and isinstance(self.transform.transforms[0].size, int) else 224, 
                                   self.transform.transforms[0].size[1] if hasattr(self.transform.transforms[0], 'size') and isinstance(self.transform.transforms[0].size, tuple) else 224)), torch.tensor(0)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class EfficientNetPipeline(BasePipeline):
    """
    Pipeline for training and evaluating EfficientNet models on the Autism dataset.
    """
    
    def __init__(self, experiment_name: str, run_name: str, data_loader: AutismDataLoader,
                 model_name: str = "efficientnet_b0", num_classes: int = 2):
        """
        Initialize the EfficientNet pipeline.
        
        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Name of this specific run
            data_loader: Instance of AutismDataLoader
            model_name: Which EfficientNet model to use (b0-b7)
            num_classes: Number of classes for classification
        """
        super().__init__(experiment_name, run_name, data_loader)
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224 # Default for EfficientNet-B0
        if self.model_name == "efficientnet_b0": input_size = 224
        elif self.model_name == "efficientnet_b1": input_size = 240
        elif self.model_name == "efficientnet_b2": input_size = 260
        elif self.model_name == "efficientnet_b3": input_size = 300
        elif self.model_name == "efficientnet_b4": input_size = 380
        elif self.model_name == "efficientnet_b5": input_size = 456
        elif self.model_name == "efficientnet_b6": input_size = 528
        elif self.model_name == "efficientnet_b7": input_size = 600
        
        self.train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def create_model(self):
        """Create and return an EfficientNet model with the specified configuration"""
        
        model_creator = getattr(models, self.model_name, None)
        if not model_creator:
            raise ValueError(f"Unknown EfficientNet variant: {self.model_name}")
        
        # Use the modern weights API if available (torchvision >= 0.13)
        try:
            weights_enum = getattr(models, f"{self.model_name.upper()}_Weights", None)
            if weights_enum:
                model = model_creator(weights=weights_enum.IMAGENET1K_V1)
            else: # Fallback for older torchvision or if specific enum doesn't exist
                model = model_creator(pretrained=True)
        except AttributeError: # Fallback for very old torchvision
             model = model_creator(pretrained=True)

        # Common way to access classifier for EfficientNet from torchvision
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            if len(model.classifier) > 0 and isinstance(model.classifier[-1], nn.Linear):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, self.num_classes)
            else:
                 raise AttributeError(f"Cannot find Linear layer in model.classifier for {self.model_name}")
        elif hasattr(model, '_fc'): # Some older torchvision EfficientNet versions use _fc
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, self.num_classes)
        else:
            raise AttributeError(f"Cannot find classifier layer for {self.model_name} (tried .classifier[-1] and ._fc)")
        return model.to(self.device)
    
    def prepare_data(self, data):
        """
        Convert the data from AutismDataLoader into PyTorch DataLoaders.
        
        Returns:
            Dict containing train_loader, val_loader, and test_loader
        """
        dataloaders = {}
        
        # Create PyTorch datasets and dataloaders for each split
        for split, split_data in data.items():
            if not split_data['file_paths']:
                print(f"Warning: No file paths for split '{split}'. Skipping loader creation.")
                continue
            transform = self.train_transform if split == 'train' else self.val_transform
            
            dataset = AutismImageDataset(
                file_paths=split_data['file_paths'],
                labels=split_data['labels'],
                transform=transform
            )
            
            if len(dataset) == 0:
                print(f"Warning: Dataset for split '{split}' is empty. Skipping loader creation.")
                continue
            
            # Use a smaller batch size if running on CPU
            batch_size = 16 if torch.cuda.is_available() else 4
            
            dataloaders[f"{split}_loader"] = TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=2 if torch.cuda.is_available() else 0,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        return dataloaders
    
    def train(self, data: Dict, **params) -> nn.Module:
        """
        Train the EfficientNet model on the provided data.
        
        Args:
            data: Dict from AutismDataLoader.load_data()
            params: Additional parameters for training
                - epochs: Number of training epochs
                - learning_rate: Learning rate for optimizer
                - weight_decay: Weight decay for optimizer
        
        Returns:
            Trained PyTorch model
        """
        # Extract parameters with defaults
        epochs = params.get('epochs', 10)
        learning_rate = params.get('learning_rate', 0.001)
        weight_decay = params.get('weight_decay', 1e-5)
        
        # Prepare data loaders
        loaders = self.prepare_data(data)
        train_loader = loaders.get('train_loader')
        val_loader = loaders.get('valid_loader')
        
        if not train_loader:
            raise ValueError("No training data available")
        
        # Create model
        model = self.create_model()
        print(f"Training on device: {self.device}")
        print(f"Training {self.model_name} for {epochs} epochs")
        print(f"Dataset sizes: Train={len(train_loader.dataset)} samples", end="")
        if val_loader:
            print(f", Validation={len(val_loader.dataset)} samples")
        else:
            print("")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training loop
        best_val_acc = 0.0
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Progress bar for batches
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                batch_loss = loss.item()
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                pbar.set_postfix({'loss': f"{batch_loss:.4f}", 'acc': f"{batch_acc:.4f}"})
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Log training metrics for this epoch
            metrics = {
                f"train_loss_epoch_{epoch}": train_loss,
                f"train_acc_epoch_{epoch}": train_acc
            }
            
            # Validation phase if validation data is available
            if val_loader:
                print(f"Validating...")
                val_metrics = self._evaluate_split(model, val_loader, criterion)
                metrics.update({
                    f"val_loss_epoch_{epoch}": val_metrics["loss"],
                    f"val_acc_epoch_{epoch}": val_metrics["accuracy"]
                })
                
                # Save best model based on validation accuracy
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    print(f"✓ New best validation accuracy: {best_val_acc:.4f}")
            
            # Log metrics for this epoch
            for key, value in metrics.items():
                self.log_metrics({key: value})
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if val_loader:
                print(f"  Valid Loss: {val_metrics['loss']:.4f}, Valid Acc: {val_metrics['accuracy']:.4f}")
            print("-" * 60)
        
        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        return model
    
    def _evaluate_split(self, model: nn.Module, loader: TorchDataLoader, 
                        criterion: nn.Module) -> Dict[str, float]:
        """
        Evaluate the model on a specific data split.
        
        Returns:
            Dict with metrics (loss, accuracy)
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(loader, desc="Evaluating")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return {
            "loss": running_loss / total,
            "accuracy": correct / total
        }
    
    def evaluate(self, model: nn.Module, data: Dict) -> Dict[str, float]:
        """
        Evaluate the trained model using the centralized metric logging utility.
        """
        loaders = self.prepare_data(data)
        # Prefer test set for final evaluation, fallback to validation set
        eval_loader = loaders.get('test_loader')
        eval_split_name = "test"
        if not eval_loader:
            eval_loader = loaders.get('valid_loader')
            eval_split_name = "validation"
        
        if not eval_loader:
            print(f"WARNING: No test or validation data available for final evaluation of {self.model_name}. Returning empty metrics.")
            return {}

        print(f"Starting final evaluation of {self.model_name} on '{eval_split_name}' set...")
        all_preds_list, all_labels_list, all_probs_list = [], [], []
        model.eval()
        with torch.no_grad():
            pbar_final_eval = tqdm(eval_loader, desc=f"Final Eval ({self.model_name} on {eval_split_name})")
            for inputs, labels_batch in pbar_final_eval:
                inputs_device = inputs.to(self.device)
                outputs = model(inputs_device)
                # Handle potential tuple output from models like InceptionV3 if used here by mistake
                if isinstance(outputs, tuple):
                     outputs = outputs[0] # Assuming primary output is first
                
                probs_batch = torch.softmax(outputs, dim=1)
                _, preds_batch = torch.max(outputs, 1)
                
                all_preds_list.extend(preds_batch.cpu().numpy())
                all_labels_list.extend(labels_batch.cpu().numpy())
                all_probs_list.extend(probs_batch.cpu().numpy())
        
        if not all_labels_list: # Check if any data was processed
            print(f"WARNING: No data processed during final evaluation for {self.model_name}. Returning empty metrics.")
            return {}
            
        all_labels_np = np.array(all_labels_list)
        all_preds_np = np.array(all_preds_list)
        all_probs_np = np.array(all_probs_list)
        
        class_names_for_eval = ["Non_Autistic", "Autistic"] if self.num_classes == 2 else [f"class_{i}" for i in range(self.num_classes)]
        metric_prefix_final = "eval" # Standard prefix for final evaluation metrics

        print(f"Logging standard classification metrics for {self.model_name} with prefix '{metric_prefix_final}'...")
        final_metrics = log_standard_classification_metrics(
            y_true=all_labels_np,
            y_pred=all_preds_np,
            class_names=class_names_for_eval,
            metric_prefix=metric_prefix_final,
            num_classes=self.num_classes,
            y_probs=all_probs_np
        )
        
        print(f"\nFinal Evaluation Metrics for {self.model_name} (from utility):")
        for k, v_val in final_metrics.items():
            if "plot_path" not in k: # Don't try to print path as float
                print(f"  {k}: {v_val:.4f}")
        print("-" * 60)
            
        return final_metrics # This dict is returned to BasePipeline which calls self.log_metrics()

    def run(self, **params) -> mlflow.entities.Run:
        """
        Execute the pipeline: load data, train model, evaluate, and log to MLflow.
        """
        # Load data
        data = self.data_loader.load_data()

        with mlflow.start_run(run_name=self.run_name) as run:
            # Log parameters
            self.log_params(params)
            # Log model name as a tag
            model_name = params.get('model_name', self.model_name)
            mlflow.set_tag("model_name", model_name)
            # Train the model
            model = self.train(data, **params)
            # Evaluate the model
            metrics = self.evaluate(model, data)
            # Log metrics
            self.log_metrics(metrics)
            # Log model artifacts
            self.log_model(model)
            return run 