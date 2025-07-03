"""
Classe para orquestrar experimentos de treinamento usando PyTorch.
"""

import os
import time
import yaml
from typing import Dict, Any, List, Union
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from data_handler import DataHandler, AlbumentationsDataset, AlbumentationsDatasetWithFeatures
from model_factory import ModelFactory
from evaluator import ModelEvaluator
from sklearn.model_selection import StratifiedKFold
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str ='.') -> Dict[str, Any]:
    """Achatamento de dicionário aninhado para logging no MLflow."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AlbumentationsSubset(torch.utils.data.Dataset):
    """Subset que permite usar transformações diferentes em subsets (útil para K-Fold)."""
    def __init__(self, parent_dataset, indices, transform):
        self.parent_dataset = parent_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        result = self.parent_dataset[real_idx]
        # Aplica transformações se definidas
        if self.transform:
            if isinstance(result, tuple):
                if len(result) == 3:
                    image, feats, label = result
                    image = self.transform(image=image)["image"]
                    return image, feats, label
                elif len(result) == 2:
                    image, label = result
                    image = self.transform(image=image)["image"]
                    return image, label
        return result


class Trainer:
    """Classe Trainer simplificada para compatibilidade com main.py"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.experiment_trainer = ExperimentTrainer(config)
        
    def train_model(self, model, model_name: str) -> Dict[str, Any]:
        results = self.experiment_trainer.run_experiments([model_name])
        if model_name in results and 'error' not in results[model_name]:
            return results[model_name]
        raise Exception(f"Erro no treinamento de {model_name}: {results.get(model_name, {}).get('error', 'Erro desconhecido')}")


class EarlyStopping:
    """Callback para Early Stopping."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model: nn.Module):
        self.best_weights = model.state_dict().copy()


class ExperimentTrainer:
    """Classe responsável por orquestrar experimentos de treinamento."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎮 Usando dispositivo: {self.device}")
        
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        if 'tracking_uri' in config['mlflow']:
            mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        
        self.data_handler = DataHandler(config['data'])
        
        # Número de folds para K-Fold cross-validation (1 = desativado)
        self.k_folds = config.get('training', {}).get('k_folds', 1)
        
        print(f"✅ ExperimentTrainer inicializado")
        print(f"   Dispositivo: {self.device}")
        print(f"   Dataset: {config['data']['dataset_path']}")
        print(f"   Experimento MLflow: {config['mlflow']['experiment_name']}")
    
    def run_experiments(self, model_list: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        print(f"🚀 Iniciando experimentos para {len(model_list)} modelos: {', '.join(model_list)}")
        
        for i, model_name in enumerate(model_list, 1):
            print(f"\n{'='*60}\n📋 EXPERIMENTO {i}/{len(model_list)}: {model_name}\n{'='*60}")
            
            try:
                if self.k_folds > 1:
                    print(f"🔄 Executando {self.k_folds}-Fold Cross-Validation devido ao conjunto de validação pequeno…")
                    result = self._run_single_experiment_kfold(model_name)
                else:
                    result = self._run_single_experiment(model_name)
                results[model_name] = result
                print(f"✅ {model_name} concluído com sucesso! Acurácia final: {result.get('accuracy', 0):.4f}")
            except Exception as e:
                print(f"❌ Erro no experimento {model_name}: {e}")
                results[model_name] = {'error': str(e)}
                
        return results
    
    def _run_single_experiment(self, model_name: str) -> Dict[str, Any]:
        model_config = self._build_model_config(model_name)
        
        train_loader, val_loader, test_loader, num_classes = self.data_handler.get_data_loaders(model_config)
        
        model = ModelFactory.create_model(
            model_name=model_name,
            input_shape=(self.config['data']['image_height'], self.config['data']['image_width'], self.config['data']['channels']),
            num_classes=num_classes,
            config=model_config
        ).to(self.device)
        
        with mlflow.start_run(run_name=f"{model_name}_experiment"):
            dataset_info = {
                'train_samples': len(train_loader.dataset),
                'val_samples': len(val_loader.dataset),
                'test_samples': len(test_loader.dataset),
            }
            self._log_parameters(model_name, model_config, num_classes, dataset_info)
            
            history = self._train_model(model, train_loader, val_loader, model_config)
            
            results = self._evaluate_model(model, test_loader, model_name)
            
            self._log_results(results, model, model_name, history)
            
            return results
    
    def _train_model(self, model: nn.Module, train_loader, val_loader, 
                    model_config: Dict[str, Any]) -> Dict[str, List[float]]:
        print(f"Iniciando treinamento...")
        
        optimizer = self._setup_optimizer(model, model_config)
        criterion = nn.CrossEntropyLoss()
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min',
            factor=model_config.get('reduce_lr_factor', 0.5),
            patience=model_config.get('reduce_lr_patience', 5),
            min_lr=model_config.get('min_lr', 1e-7)
        )
        
        early_stopping = EarlyStopping(
            patience=model_config.get('early_stopping_patience', 10),
            min_delta=model_config.get('early_stopping_min_delta', 0.001),
            restore_best_weights=True
        )
        
        monitor_metric = model_config.get('early_stopping_monitor', 'val_acc')  # 'val_loss' ou 'val_acc'
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        epochs = model_config.get('epochs', 30)
        
        # Configuração de unfreeze gradual: lista de épocas onde novas camadas devem ser
        # liberadas para treinamento. Ex.: [5, 15] – na época 5 libera parte das camadas,
        # na 15 libera o restante. Pode ser definida no bloco `training` do config.yaml
        # como `gradual_unfreeze_milestones`. Se não definido, nenhuma camada adicional é
        # liberada.
        unfreeze_milestones = model_config.get('gradual_unfreeze_milestones', [])
        
        for epoch in range(epochs):
            print(f"\n📅 Época {epoch+1}/{epochs}")
            
            # ---------------------------------------------------------
            # Gradual Unfreeze
            # ---------------------------------------------------------
            if epoch in unfreeze_milestones:
                self._apply_gradual_unfreeze(model, epoch, len(unfreeze_milestones), model_config)
                # Após liberar parâmetros precisamos recriar otimizador e scheduler
                optimizer = self._setup_optimizer(model, model_config)
                scheduler = ReduceLROnPlateau(
                    optimizer, mode='min',
                    factor=model_config.get('reduce_lr_factor', 0.5),
                    patience=model_config.get('reduce_lr_patience', 5),
                    min_lr=model_config.get('min_lr', 1e-7)
                )
            
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            scheduler.step(val_loss)
            
            for k, v in {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}.items():
                history[k].append(v)
            
            mlflow.log_metrics({
                'train_loss': train_loss, 'train_accuracy': train_acc,
                'val_loss': val_loss, 'val_accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Escolhe métrica de parada antecipada
            metric_for_es = (-val_loss if monitor_metric == 'val_loss' else val_acc)

            if early_stopping(metric_for_es, model):
                print(f"🛑 Early stopping na época {epoch+1}")
                break
        
        print(f"✅ Treinamento concluído!")
        return history
    
    def _setup_optimizer(self, model: nn.Module, model_config: Dict[str, Any]) -> optim.Optimizer:
        optimizer_name = model_config.get('optimizer', 'adam').lower()
        learning_rate = model_config.get('initial_learning_rate', 0.001)
        weight_decay = model_config.get('weight_decay', 1e-4)
        momentum = model_config.get('momentum', 0.9)
        
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamax':
            return optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            # Weight decay default 0.01 se não especificado
            if 'weight_decay' not in model_config:
                weight_decay = 0.01
            return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Otimizador não suportado: {optimizer_name}")
    
    def _train_epoch(self, model: nn.Module, train_loader, criterion, optimizer) -> tuple:
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Detecta se o batch tem features extras (3 tensores) ou só imagem (2 tensores)
            if len(batch) == 3:
                data, feats, target = batch
                data, feats, target = data.to(self.device), feats.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data, feats)
            else:
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"     Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
        
        return running_loss / len(train_loader), correct / total
    
    def _validate_epoch(self, model: nn.Module, val_loader, criterion) -> tuple:
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Detecta se o batch tem features extras (3 tensores) ou só imagem (2 tensores)
                if len(batch) == 3:
                    data, feats, target = batch
                    data, feats, target = data.to(self.device), feats.to(self.device), target.to(self.device)
                    output = model(data, feats)
                else:
                    data, target = batch
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                
                loss = criterion(output, target)
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return running_loss / len(val_loader), correct / total
    
    def _evaluate_model(self, model: nn.Module, test_loader, model_name: str) -> Dict[str, Any]:
        print(f"\n📊 Avaliando modelo {model_name}...")
        evaluator = ModelEvaluator(model=model, test_dataloader=test_loader, class_names=self.data_handler.class_names, device=self.device)
        return evaluator.evaluate()
    
    def _log_parameters(self, model_name: str, model_config: Dict[str, Any], num_classes: int, dataset_info: Dict[str, int]):
        """Registra um conjunto expandido de parâmetros no MLflow."""
        # Achata dicionários para logging
        params_to_log = {
            "model_name": model_name,
            "num_classes": num_classes,
            **_flatten_dict(self.config.get('data', {}), 'data'),
            **_flatten_dict(self.config.get('training', {}), 'training'),
            **_flatten_dict(model_config, 'model'),
            **_flatten_dict(dataset_info, 'dataset_info')
        }
        
        # Adiciona info do ambiente
        params_to_log['env.device'] = str(self.device)
        params_to_log['env.pytorch_version'] = torch.__version__
        if torch.cuda.is_available():
            params_to_log['env.cuda_version'] = torch.version.cuda
            params_to_log['env.gpu_name'] = torch.cuda.get_device_name(0)
            
        mlflow.log_params(params_to_log)
        print("📝 Parâmetros detalhados logados no MLflow.")

    def _log_results(self, results: Dict[str, Any], model: nn.Module, 
                    model_name: str, history: Dict[str, List[float]]):
        """Registra um conjunto expandido de resultados no MLflow."""
        # Log de métricas finais
        final_metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(final_metrics)
        
        # Log do modelo
        mlflow.pytorch.log_model(model, f"{model_name}_model")
        
        # Salva e loga modelo localmente (para backup)
        os.makedirs('artifacts', exist_ok=True)
        model_path = f'artifacts/{model_name}_best.pth'
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path, "model")
        
        # Log de artefatos
        for key in ['confusion_matrix_path', 'classification_report_path', 'predictions_path']:
            if key in results and os.path.exists(results[key]):
                mlflow.log_artifact(results[key], "evaluation")
        
        # Salva e loga histórico de treinamento
        history_df = pd.DataFrame(history)
        history_path = f'artifacts/{model_name}_history.csv'
        history_df.to_csv(history_path, index=False)
        mlflow.log_artifact(history_path, "history")
        
        print("📁 Artefatos (métricas, modelo, histórico, plots) salvos e logados no MLflow.")

    def _run_single_experiment_kfold(self, model_name: str) -> Dict[str, Any]:
        """Executa treinamento e avaliação usando K-Fold Cross-Validation."""
        model_config = self._build_model_config(model_name)

        with mlflow.start_run(run_name=f"{model_name}_kfold_experiment"):
            # Obter transforms (reaproveita lógica do DataHandler)
            train_t, test_t = self.data_handler.get_albumentations_transforms(model_config)

            # Definir diretório com todos os dados de treino/validação
            consolidated_dir = os.path.join(self.data_handler.dataset_path, 'consolidated')
            if os.path.exists(consolidated_dir):
                data_dir = consolidated_dir
            else:
                data_dir = os.path.join(self.data_handler.dataset_path, 'train')
                if not os.path.exists(data_dir):
                    # fallback para dataset raiz
                    data_dir = self.data_handler.dataset_path

            # Dataset completo (sem transform para aplicar transform no subset)
            # Verificar se há features disponíveis
            features_df = None
            if hasattr(self.data_handler, 'features_csv') and self.data_handler.features_csv:
                if os.path.exists(self.data_handler.features_csv):
                    print(f"📑 Carregando features para K-Fold: {self.data_handler.features_csv}")
                    features_df = pd.read_csv(self.data_handler.features_csv)
            
            if features_df is not None:
                full_dataset = AlbumentationsDatasetWithFeatures(data_dir=data_dir, features_df=features_df, transform=None)
                # Ajustar config: número de features
                model_config['num_tabular_features'] = len([c for c in features_df.columns if c not in ('image_path', 'class')])
            else:
                full_dataset = AlbumentationsDataset(data_dir=data_dir, transform=None)
            labels = full_dataset.labels

            # Test loader permanece o mesmo carregado pelo DataHandler padrão
            _, _, test_loader, num_classes = self.data_handler.get_data_loaders(model_config)

            skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.config['data']['seed'])

            # Log parâmetros apenas uma vez (antes do primeiro fold)
            dataset_info = self.data_handler.load_dataset()
            self._log_parameters(model_name, model_config, 2, dataset_info)  # num_classes=2 assumido

            fold_metrics = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                print(f"\n🔀 Fold {fold+1}/{self.k_folds}")

                train_subset = AlbumentationsSubset(full_dataset, train_idx, train_t)
                val_subset = AlbumentationsSubset(full_dataset, val_idx, test_t)

                num_workers = 0 if os.name == 'nt' else 4

                train_loader = torch.utils.data.DataLoader(
                    train_subset, batch_size=self.data_handler.batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=torch.cuda.is_available())

                val_loader = torch.utils.data.DataLoader(
                    val_subset, batch_size=self.data_handler.batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=torch.cuda.is_available())

                # Criar modelo
                model = ModelFactory.create_model(
                    model_name=model_name,
                    input_shape=(self.config['data']['image_height'], self.config['data']['image_width'], self.config['data']['channels']),
                    num_classes=num_classes,
                    config=model_config
                ).to(self.device)

                # Treinar
                history = self._train_model(model, train_loader, val_loader, model_config)

                # Avaliar
                fold_name = f"{model_name}_fold_{fold+1}"
                results = self._evaluate_model(model, test_loader, fold_name)
                # Log artifacts & metrics por fold
                self._log_results(results, model, fold_name, history)
                fold_metrics.append(results)

            # Agregar métricas
            aggregated = {}
            for key in fold_metrics[0]:
                if isinstance(fold_metrics[0][key], (int, float)):
                    aggregated[key] = float(np.mean([fm[key] for fm in fold_metrics]))

            print(f"\n📈 Métricas médias após {self.k_folds} folds: {aggregated}")
            mlflow.log_metrics({f"cv_{k}": v for k, v in aggregated.items()})

            return aggregated

    # ------------------------------------------------------------------
    def _apply_gradual_unfreeze(self, model: nn.Module, epoch: int, total_stages: int, model_config: Dict[str, Any]):
        """Libera gradualmente camadas congeladas.

        Estratégia simples: na primeira ocorrência libera 50 % dos parâmetros ainda
        congelados; na última occurrence libera todos os restantes.
        """
        frozen_params = [p for p in model.parameters() if not p.requires_grad]
        if not frozen_params:
            return  # nada a fazer

        milestones = model_config.get('gradual_unfreeze_milestones', [])
        is_last_stage = (epoch == milestones[-1]) if milestones else True

        if is_last_stage:
            # Libera tudo
            for p in frozen_params:
                p.requires_grad = True
            print(f"🔓 [GradualUnfreeze] Todas as camadas liberadas para treino na época {epoch}.")
        else:
            # Libera ~25 % das ainda congeladas
            n_to_unfreeze = max(1, len(frozen_params) // 4)
            for p in frozen_params[-n_to_unfreeze:]:
                p.requires_grad = True
            print(f"🔓 [GradualUnfreeze] Liberadas {n_to_unfreeze} de {len(frozen_params)} camadas congeladas.")

    # ------------------------------------------------------------------
    def _build_model_config(self, model_name: str) -> Dict[str, Any]:
        """Combina defaults do modelo, bloco models.<name> e bloco training."""
        # Defaults fornecidos pelo próprio modelo
        base_cfg = ModelFactory.get_model_config(model_name)

        # Config específico no YAML, se existir
        yaml_models_block = self.config.get('models', {})
        yaml_model_cfg = {}
        for key, val in yaml_models_block.items():
            # chaves podem vir em lowercase; normaliza
            if key.lower() == model_name.lower():
                yaml_model_cfg = val or {}
                break

        # Bloco global training
        global_training_cfg = self.config.get('training', {})

        # Ordem de prioridade: defaults < yaml_model_cfg < global_training_cfg
        merged = {**base_cfg, **yaml_model_cfg, **global_training_cfg}
        return merged