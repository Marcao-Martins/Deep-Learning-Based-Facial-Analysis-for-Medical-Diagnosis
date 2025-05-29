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

# Assuming my_mlflow_utils is in PYTHONPATH or accessible
# Add project root to sys.path in train_resnet.py if necessary
from my_mlflow_utils import BasePipeline
from my_mlflow_utils.data_loader import AutismDataLoader # Or specific loader if different
from my_mlflow_utils.mlflow_utils import log_standard_classification_metrics

class AutismImageDataset(Dataset):
    """PyTorch Dataset for loading Autism dataset images."""
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
            # Attempt to get size from transform if available for placeholder
            h, w = (224, 224) # Default ResNet50 input size
            if self.transform and hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Resize) and isinstance(t.size, (list, tuple)):
                        h, w = t.size[0], t.size[1]
                        break
                    elif isinstance(t, transforms.Resize) and isinstance(t.size, int):
                        h,w = t.size, t.size # If Resize is int, it applies to smaller edge
                        break 
            return torch.zeros((3, h, w)), torch.tensor(0) 
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class ResNetPipeline(BasePipeline):
    """
    Pipeline for training and evaluating ResNet-50 on the Autism dataset,
    based on the article's approach.
    """
    def __init__(self, experiment_name: str, run_name: str, data_loader: AutismDataLoader,
                 num_classes: int = 2, model_name_tag: str = "ResNet50"):
        super().__init__(experiment_name, run_name, data_loader)
        self.num_classes = num_classes
        self.model_architecture = "resnet50" # To select ResNet-50
        self.model_name_tag = model_name_tag # For MLflow tag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Standard ImageNet normalization and ResNet-50 input size (224x224)
        input_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # As per article: resizing, normalization. Augmentations like ColorJitter can be added.
        # The article mentions "image enhancement and denoising" as preprocessing.
        # Standard augmentations are a form of enhancement. Denoising is more complex and dataset-specific.
        # We will start with standard augmentations.
        self.train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), # Added rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Added ColorJitter
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        print(f"ResNetPipeline initialized for {self.model_architecture}. Device: {self.device}")

    def create_model(self) -> nn.Module:
        print(f"Creating model: {self.model_architecture}")
        # Using ResNet-50 as specified in the article
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all parameters first (transfer learning)
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer (fc)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Parameters of the new fc layer should be trainable
        for param in model.fc.parameters():
            param.requires_grad = True
            
        return model.to(self.device)

    def prepare_data(self, data: Dict[str, Dict[str, List[Any]]]) -> Dict[str, TorchDataLoader]:
        dataloaders = {}
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
                print(f"Warning: Dataset for split '{split}' is empty after transform. Skipping loader.")
                continue

            batch_size = 16 if torch.cuda.is_available() else 8 # Adjusted for potential memory
            dataloaders[f"{split}_loader"] = TorchDataLoader(
                dataset, batch_size=batch_size, shuffle=(split == 'train'),
                num_workers=2 if torch.cuda.is_available() else 0,
                pin_memory=True if torch.cuda.is_available() else False
            )
        return dataloaders

    def _evaluate_split(self, model: nn.Module, loader: TorchDataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Simplified evaluation for epoch-level validation (loss, accuracy), console output only."""
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            pbar_eval = tqdm(loader, desc=f"Epoch Val ({self.model_architecture})")
            for inputs, labels in pbar_eval:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar_eval.set_postfix({'loss': f"{loss.item():.4f}"})
        return {
            "loss": running_loss / total if total > 0 else 0,
            "accuracy": correct / total if total > 0 else 0
        }

    def train(self, data: Dict, **params) -> nn.Module:
        fc_epochs = params.get('fc_epochs', 10) # Epochs for FC layer training
        fine_tune_epochs = params.get('fine_tune_epochs', 0) # Epochs for fine-tuning deeper layers
        initial_lr = params.get('learning_rate', 0.001) # LR for FC layer
        fine_tune_lr = params.get('fine_tune_lr', 0.0001) # LR for fine-tuning
        weight_decay = params.get('weight_decay', 1e-4)
        num_unfreeze_blocks = params.get('num_unfreeze_blocks', 0) # 0: FC only, 1: layer4, 2: layer4+layer3

        loaders = self.prepare_data(data)
        train_loader = loaders.get('train_loader')
        val_loader = loaders.get('valid_loader')
        if not train_loader:
            raise ValueError("No training data available after prepare_data.")
        
        model = self.create_model()
        mlflow.set_tag("model_architecture", self.model_architecture)
        mlflow.set_tag("model_family", self.model_name_tag)
        if num_unfreeze_blocks > 0 and fine_tune_epochs > 0:
            mlflow.set_tag("fine_tuning_strategy", f"Unfreeze_{num_unfreeze_blocks}_blocks_after_{fc_epochs}_epochs")
        else:
            mlflow.set_tag("fine_tuning_strategy", "FC_only")


        criterion = nn.CrossEntropyLoss()
        
        total_epochs = fc_epochs + fine_tune_epochs
        print(f"Starting training for {self.model_name_tag} on {self.device}.")
        print(f"FC layer training: {fc_epochs} epochs, LR: {initial_lr}")
        if fine_tune_epochs > 0 and num_unfreeze_blocks > 0:
            print(f"Fine-tuning: {fine_tune_epochs} epochs, LR: {fine_tune_lr}, Unfreezing {num_unfreeze_blocks} block(s).")
        else:
            print("Fine-tuning deeper layers is disabled.")

        # Initial optimizer for FC layer only
        optimizer = optim.Adam(model.fc.parameters(), lr=initial_lr, weight_decay=weight_decay)
        
        best_val_acc = 0.0

        for epoch in range(total_epochs):
            current_stage = "FC_Train"
            # Check if it's time to switch to fine-tuning
            if epoch == fc_epochs and fine_tune_epochs > 0 and num_unfreeze_blocks > 0:
                print(f"--- Epoch {epoch+1}/{total_epochs}: Unfreezing layers for fine-tuning ---")
                current_stage = "FineTune_Train"
                
                layers_to_unfreeze_names = []
                if num_unfreeze_blocks >= 1 and hasattr(model, 'layer4'):
                    for param in model.layer4.parameters():
                        param.requires_grad = True
                    layers_to_unfreeze_names.append("layer4")
                if num_unfreeze_blocks >= 2 and hasattr(model, 'layer3'):
                    for param in model.layer3.parameters():
                        param.requires_grad = True
                    layers_to_unfreeze_names.append("layer3")
                if num_unfreeze_blocks >= 3 and hasattr(model, 'layer2'): # Example for more blocks
                    for param in model.layer2.parameters():
                        param.requires_grad = True
                    layers_to_unfreeze_names.append("layer2")
                
                print(f"Unfrozen layers: {', '.join(layers_to_unfreeze_names) if layers_to_unfreeze_names else 'None'}")
                
                # Optimizer for fine-tuning: includes FC and newly unfrozen layers
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=fine_tune_lr, 
                    weight_decay=weight_decay
                )
                print(f"Optimizer updated for fine-tuning with LR: {fine_tune_lr}")
            elif epoch < fc_epochs:
                 current_stage = "FC_Train"
            else: # This is during fine_tune_epochs, using the fine-tune optimizer
                 current_stage = "FineTune_Train"


            model.train()
            running_loss, correct, total = 0.0, 0, 0
            desc_prefix = f"Epoch {epoch+1}/{total_epochs} [{self.model_name_tag} {current_stage}]"
            pbar_train = tqdm(train_loader, desc=desc_prefix)
            for inputs, labels in pbar_train:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar_train.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(predicted == labels).sum().item()/labels.size(0):.4f}"})
            
            train_loss = running_loss / total if total > 0 else 0
            train_acc = correct / total if total > 0 else 0
            # No MLflow logging for epoch-level train metrics to keep MLflow UI clean

            val_epoch_loss, val_epoch_acc = -1.0, -1.0
            if val_loader:
                val_metrics_split = self._evaluate_split(model, val_loader, criterion)
                val_epoch_loss = val_metrics_split["loss"]
                val_epoch_acc = val_metrics_split["accuracy"]
                # No MLflow logging for epoch-level val metrics
                if val_epoch_acc > best_val_acc:
                    best_val_acc = val_epoch_acc
                    print(f"✓ New best validation accuracy for {self.model_name_tag} ({current_stage}): {best_val_acc:.4f}")
            
            print(f"Epoch {epoch+1}/{total_epochs} [{self.model_name_tag} {current_stage}] Summary - Train L: {train_loss:.4f}, A: {train_acc:.4f} | Val L: {val_epoch_loss:.4f}, A: {val_epoch_acc:.4f}")
        
        print(f"Training for {self.model_name_tag} completed. Best validation accuracy (local): {best_val_acc:.4f}")
        return model

    def evaluate(self, model: nn.Module, data: Dict) -> Dict[str, Any]:
        loaders = self.prepare_data(data)
        eval_loader = loaders.get('test_loader')
        eval_split_name = "test"
        if not eval_loader:
            eval_loader = loaders.get('valid_loader')
            eval_split_name = "validation"
        
        if not eval_loader:
            print(f"WARNING: No test or validation data for final eval of {self.model_name_tag}. Returning empty.")
            return {}

        print(f"Starting final evaluation of {self.model_name_tag} on '{eval_split_name}' set...")
        all_preds_list, all_labels_list, all_probs_list = [], [], []
        model.eval()
        with torch.no_grad():
            pbar_final_eval = tqdm(eval_loader, desc=f"Final Eval ({self.model_name_tag} on {eval_split_name})")
            for inputs, labels_batch in pbar_final_eval:
                inputs_device = inputs.to(self.device)
                outputs = model(inputs_device)
                probs_batch = torch.softmax(outputs, dim=1)
                _, preds_batch = torch.max(outputs, 1)
                all_preds_list.extend(preds_batch.cpu().numpy())
                all_labels_list.extend(labels_batch.cpu().numpy())
                all_probs_list.extend(probs_batch.cpu().numpy())
        
        if not all_labels_list:
            print(f"WARNING: No data processed in final eval for {self.model_name_tag}. Returning empty.")
            return {}
            
        all_labels_np = np.array(all_labels_list)
        all_preds_np = np.array(all_preds_list)
        all_probs_np = np.array(all_probs_list)
        
        class_names_for_eval = ["Non_Autistic", "Autistic"] if self.num_classes == 2 else [f"class_{i}" for i in range(self.num_classes)]
        metric_prefix_final = "eval" # Standard prefix for final evaluation metrics

        print(f"Logging standard classification metrics for {self.model_name_tag} with prefix '{metric_prefix_final}'...")
        final_metrics = log_standard_classification_metrics(
            y_true=all_labels_np,
            y_pred=all_preds_np,
            class_names=class_names_for_eval,
            metric_prefix=metric_prefix_final,
            num_classes=self.num_classes,
            y_probs=all_probs_np
        )
        
        print(f"\nFinal Evaluation Metrics for {self.model_name_tag} (from utility with prefix '{metric_prefix_final}'):")
        for k, v_val in final_metrics.items():
            if "plot_path" not in k:
                print(f"  {k}: {v_val:.4f}")
        print("-" * 60)
            
        return final_metrics

    # log_model is inherited from BasePipeline and should work by default for PyTorch models.
    # No _log_confusion_matrix needed here as it's handled by the utility. 