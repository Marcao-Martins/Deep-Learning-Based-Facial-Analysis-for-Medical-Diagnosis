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

# Assuming my_mlflow_utils is in the parent directory or PYTHONPATH
from my_mlflow_utils import BasePipeline
from my_mlflow_utils.data_loader import AutismDataLoader # We'll use this for initial loading
from my_mlflow_utils.mlflow_utils import log_standard_classification_metrics # <-- Import the new utility


class AugmentedAutismImageDataset(Dataset):
    """PyTorch Dataset for loading Autism dataset images with augmentations."""
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
            print(f"ERROR: Image file not found {img_path}")
            # Return a placeholder or raise an error, depending on desired handling
            # For now, let's return a black image and a default label
            return torch.zeros((3, 224, 224)), torch.tensor(0)


        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class EnsemblePipeline(BasePipeline):
    """
    Pipeline for training and evaluating an ensemble of CNN models
    (EfficientNetB5, MobileNetV2, InceptionV3) on the Autism dataset.
    """
    def __init__(self, experiment_name: str, run_name: str, data_loader: AutismDataLoader, num_classes: int = 2):
        super().__init__(experiment_name, run_name, data_loader) # data_loader is AutismDataLoader instance
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_names = ["EfficientNetB5", "MobileNetV2", "InceptionV3"]
        self.trained_models: Dict[str, nn.Module] = {}

        # ImageNet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Input sizes expected by models
        self.input_sizes = {
            "EfficientNetB5": 456, # Standard for B5
            "MobileNetV2": 224,
            "InceptionV3": 299
        }
        
        # Data augmentation and normalization for training
        self.train_transforms = {
            name: transforms.Compose([
                transforms.Resize((self.input_sizes[name], self.input_sizes[name])), # Resize first
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10, scale=(0.8, 1.2)), # Shift, Shear, Zoom
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]) for name in self.model_names
        }

        # Normalization for validation/testing
        self.val_transforms = {
            name: transforms.Compose([
                transforms.Resize((self.input_sizes[name], self.input_sizes[name])),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]) for name in self.model_names
        }
        print(f"EnsemblePipeline initialized. Device: {self.device}")

    def create_individual_model(self, model_name_str: str) -> nn.Module:
        """Create and return a pre-trained model with a new classifier head."""
        model = None
        in_features = 0

        if model_name_str == "EfficientNetB5":
            model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)
        elif model_name_str == "MobileNetV2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)
        elif model_name_str == "InceptionV3":
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
            # Handle Inception's auxiliary classifier if it exists and primary
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)
            if hasattr(model, 'AuxLogits') and model.AuxLogits is not None: # Required for training InceptionV3
                 model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, self.num_classes)
        else:
            raise ValueError(f"Unknown model_name_str: {model_name_str}")

        # Freeze convolutional base layers as per the paper
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the new classifier head
        if model_name_str == "EfficientNetB5" or model_name_str == "MobileNetV2":
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif model_name_str == "InceptionV3":
            for param in model.fc.parameters():
                param.requires_grad = True
            if hasattr(model, 'AuxLogits') and model.AuxLogits is not None:
                 for param in model.AuxLogits.fc.parameters():
                    param.requires_grad = True
        
        return model.to(self.device)

    def prepare_data(self, raw_data: Dict[str, Dict[str, List[str]]], model_name_str: str) -> Dict[str, TorchDataLoader]:
        """Convert raw data from AutismDataLoader into PyTorch DataLoaders for a specific model."""
        dataloaders = {}
        current_train_transform = self.train_transforms[model_name_str]
        current_val_transform = self.val_transforms[model_name_str]

        for split, split_data_content in raw_data.items():
            if not split_data_content['file_paths']: # Skip if a split is empty
                print(f"Warning: No file paths found for split '{split}' for model '{model_name_str}'. Skipping.")
                continue

            transform_to_apply = current_train_transform if split == 'train' else current_val_transform
            
            dataset = AugmentedAutismImageDataset(
                file_paths=split_data_content['file_paths'],
                labels=split_data_content['labels'],
                transform=transform_to_apply
            )
            
            if not dataset or len(dataset) == 0: # If dataset is empty
                print(f"Warning: Dataset for split '{split}' for model '{model_name_str}' is empty or could not be loaded. Skipping.")
                continue

            batch_size = 16 if torch.cuda.is_available() else 4 # Smaller batch size for CPU
            # For InceptionV3, during training, it outputs a tuple (output, aux_output)
            # The dataloader itself remains the same. Handling is in the training loop.
            dataloaders[f"{split}_loader"] = TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=2 if torch.cuda.is_available() else 0,
                pin_memory=True if torch.cuda.is_available() else False
            )
        return dataloaders 

    def _train_individual_model(self, model: nn.Module, model_name_str: str, 
                                train_loader: TorchDataLoader, val_loader: TorchDataLoader = None, 
                                **params) -> nn.Module:
        """Train a single model and log its epoch-wise metrics."""
        epochs = params.get(f'{model_name_str}_epochs', params.get('epochs', 10))
        learning_rate = params.get(f'{model_name_str}_lr', params.get('learning_rate', 0.001))
        weight_decay = params.get(f'{model_name_str}_weight_decay', params.get('weight_decay', 1e-5))
        
        print(f"\nTraining {model_name_str} for {epochs} epochs on {self.device}...")
        print(f"  LR: {learning_rate}, Weight Decay: {weight_decay}")
        print(f"  Train batches: {len(train_loader)}", end="")
        if val_loader:
            print(f", Valid batches: {len(val_loader)}")
        else:
            print("")
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        best_val_acc = 0.0 # Tracks best validation accuracy for local printing, not for MLflow logging

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_preds = 0
            total_samples = 0
            pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{model_name_str} Train]")
            for inputs, labels in pbar_train:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                if model_name_str == "InceptionV3" and model.training:
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels.data).item()
                total_samples += labels.size(0)
                pbar_train.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'acc': f"{(torch.sum(preds == labels.data).item() / labels.size(0)):.4f}"
                })
            
            epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0
            epoch_train_acc = correct_preds / total_samples if total_samples > 0 else 0
            
            epoch_val_loss = -1.0
            epoch_val_acc = -1.0
            if val_loader:
                val_run_loss, val_correct_preds, val_total_samples = 0.0, 0, 0
                model.eval()
                with torch.no_grad():
                    for v_inputs, v_labels in val_loader:
                        v_inputs, v_labels = v_inputs.to(self.device), v_labels.to(self.device)
                        v_outputs = model(v_inputs)
                        v_loss = criterion(v_outputs, v_labels)
                        val_run_loss += v_loss.item() * v_inputs.size(0)
                        _, v_preds = torch.max(v_outputs, 1)
                        val_correct_preds += torch.sum(v_preds == v_labels.data).item()
                        val_total_samples += v_labels.size(0)
                epoch_val_loss = val_run_loss / val_total_samples if val_total_samples > 0 else 0
                epoch_val_acc = val_correct_preds / val_total_samples if val_total_samples > 0 else 0
                
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    # This print remains for local console feedback during training
                    print(f"✓ {model_name_str}: New best val_acc: {best_val_acc:.4f} at epoch {epoch+1}")
            
            print(f"Epoch {epoch+1}/{epochs} [{model_name_str}] Summary:")
            print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
            if val_loader:
                print(f"  Valid Loss: {epoch_val_loss:.4f}, Valid Acc: {epoch_val_acc:.4f}")
            print("-" * 30)
            
        print(f"Training for {model_name_str} completed. Best val_acc (local): {best_val_acc:.4f}")
        return model

    def train(self, raw_data_from_autism_loader: Dict, **params) -> Dict[str, nn.Module]:
        mlflow.set_tag("ensemble_strategy", "SoftVoting")
        mlflow.set_tag("base_models", ", ".join(self.model_names))
        self.trained_models = {} # Initialize for the current run

        for model_name_str in self.model_names:
            mlflow.set_tag(f"{model_name_str}_training_status", "Starting Training")
            print(f"\n--- Preparing and Training: {model_name_str} ---")
            
            individual_model = self.create_individual_model(model_name_str)
            dataloaders = self.prepare_data(raw_data_from_autism_loader, model_name_str)

            train_loader = dataloaders.get('train_loader')
            val_loader = dataloaders.get('valid_loader')

            if not train_loader:
                print(f"ERROR: No training data loader available for {model_name_str}. Skipping training.")
                mlflow.set_tag(f"{model_name_str}_status", "Skipped - No Data")
                continue
            
            trained_individual_model = self._train_individual_model(
                model=individual_model, 
                model_name_str=model_name_str,
                train_loader=train_loader, 
                val_loader=val_loader, 
                **params
            )
            self.trained_models[model_name_str] = trained_individual_model
            mlflow.set_tag(f"{model_name_str}_status", "Training Complete")

            print(f"Logging {model_name_str} to MLflow artifacts...")
            try:
                mlflow.pytorch.log_model(trained_individual_model, artifact_path=f"model_{model_name_str}")
                print(f"{model_name_str} logged successfully as artifact.")
            except Exception as e:
                print(f"Error logging {model_name_str} with mlflow.pytorch: {e}")

        if not self.trained_models:
            raise RuntimeError("Ensemble training failed: No models were successfully trained.")
        
        print("\nAll individual models trained and artifact-logged.")
        return self.trained_models

    def evaluate(self, trained_models_dict: Dict[str, nn.Module], 
                 raw_data_from_autism_loader: Dict) -> Dict[str, float]:
        print("\n--- Evaluating Ensemble with Soft Voting ---")
        if not trained_models_dict:
            print("ERROR: No trained models available for ensemble evaluation.")
            return {}

        test_split_name = 'test'
        if test_split_name not in raw_data_from_autism_loader or not raw_data_from_autism_loader[test_split_name]['file_paths']:
            print(f"'{test_split_name}' split not found or empty. Trying 'valid' split for ensemble evaluation.")
            test_split_name = 'valid'
            if test_split_name not in raw_data_from_autism_loader or not raw_data_from_autism_loader[test_split_name]['file_paths']:
                print(f"Neither 'test' nor 'valid' split found/empty. Cannot evaluate ensemble.")
                return {}
        
        test_split_data = raw_data_from_autism_loader[test_split_name]
        print(f"Using '{test_split_name}' split for final ensemble evaluation.")

        all_model_probs_collector: List[np.ndarray] = []
        all_labels_collector: List[int] = []
        first_model_processed_labels = False

        for model_idx, (model_name_str, model_instance) in enumerate(trained_models_dict.items()):
            print(f"Getting predictions from: {model_name_str} for ensemble evaluation")
            model_instance.eval().to(self.device)
            
            current_model_transform = self.val_transforms[model_name_str]
            test_dataset_ensemble = AugmentedAutismImageDataset(
                test_split_data['file_paths'], 
                test_split_data['labels'], 
                current_model_transform
            )
            if not test_dataset_ensemble or len(test_dataset_ensemble) == 0:
                print(f"Could not create dataset for {model_name_str} on {test_split_name} split for ensemble. Skipping model.")
                continue

            bs = 16 if torch.cuda.is_available() else 4
            current_test_loader = TorchDataLoader(
                test_dataset_ensemble, batch_size=bs, shuffle=False, 
                num_workers=2 if torch.cuda.is_available() else 0, 
                pin_memory=torch.cuda.is_available()
            )
            
            current_model_probs_list: List[np.ndarray] = []
            with torch.no_grad():
                pbar = tqdm(current_test_loader, desc=f"Ensemble Probs ({model_name_str} on {test_split_name})")
                for inputs, labels_batch in pbar:
                    outputs = model_instance(inputs.to(self.device))
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    current_model_probs_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
                    if not first_model_processed_labels:
                        all_labels_collector.extend(labels_batch.cpu().numpy())
            
            if not current_model_probs_list:
                print(f"No probabilities collected for {model_name_str} for ensemble. Skipping.")
                continue
            
            all_model_probs_collector.append(np.concatenate(current_model_probs_list, axis=0))
            if not first_model_processed_labels:
                first_model_processed_labels = True
        
        if not all_model_probs_collector or not all_labels_collector:
            print("Insufficient data for ensemble eval (no probabilities or no labels). Skipping.")
            return {}

        min_samples = min(p.shape[0] for p in all_model_probs_collector)
        if len(all_labels_collector) != min_samples:
             print(f"Warning: Label count ({len(all_labels_collector)}) and min probability samples ({min_samples}) mismatch. Truncating labels.")
             all_labels_np = np.array(all_labels_collector[:min_samples])
             processed_probs_for_stacking = [probs[:min_samples] for probs in all_model_probs_collector]
        else:
            all_labels_np = np.array(all_labels_collector)
            processed_probs_for_stacking = all_model_probs_collector

        if not processed_probs_for_stacking:
            print("No probabilities available after processing. Cannot evaluate ensemble.")
            return {}
        
        stacked_probs = np.stack(processed_probs_for_stacking, axis=0)
        avg_probs = np.mean(stacked_probs, axis=0)
        ensemble_preds = np.argmax(avg_probs, axis=1)
        
        class_names_for_eval = ["Non_Autistic", "Autistic"] if self.num_classes == 2 else [f"class_{i}" for i in range(self.num_classes)]
        metric_prefix_final_ensemble = "eval" 
        
        print(f"Logging standard metrics for ENSEMBLE with prefix '{metric_prefix_final_ensemble}'...")
        final_ensemble_metrics = log_standard_classification_metrics(
            y_true=all_labels_np,
            y_pred=ensemble_preds,
            class_names=class_names_for_eval,
            metric_prefix=metric_prefix_final_ensemble,
            num_classes=self.num_classes,
            y_probs=avg_probs
        )
        
        print(f"\nFinal ENSEMBLE Evaluation Metrics (from utility with prefix '{metric_prefix_final_ensemble}'):")
        for k,v in final_ensemble_metrics.items(): 
            if "plot_path" not in k: print(f"  {k}: {v:.4f}")
        print("-"*60)
        return final_ensemble_metrics

    def log_model(self, trained_models_dict: Dict[str, nn.Module]):
        print("EnsemblePipeline.log_model called. Individual PyTorch models logged during training.")
        print("Logging ensemble configuration metadata.")
        ensemble_info = {
            "ensemble_strategy": "SoftVoting",
            "constituent_models_paths": [f"model_{name}" for name in trained_models_dict.keys()],
            "num_classes": self.num_classes,
            "notes": "This run combined predictions from individually trained models. Artifacts for each model are in model_<name> folders."
        }
        mlflow.log_dict(ensemble_info, "ensemble_configuration.json")
        mlflow.set_tag("ensemble_config_logged", "True")

# ... (End of class) ... 