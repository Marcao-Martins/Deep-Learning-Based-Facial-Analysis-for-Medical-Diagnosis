"""
Classe para avaliação de modelos usando PyTorch.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score, classification_report
)
from typing import Dict, List, Tuple, Any
import pandas as pd


class ModelEvaluator:
    """Classe para avaliação abrangente de modelos."""
    
    def __init__(self, model: nn.Module,
                 test_dataloader: DataLoader,
                 class_names: List[str],
                 device: torch.device = None):
        """
        Inicializa o avaliador.
        
        Args:
            model: Modelo PyTorch a ser avaliado
            test_dataloader: DataLoader do conjunto de teste
            class_names: Lista com nomes das classes
            device: Dispositivo (CPU/GPU) para computação
        """
        self.model = model
        self.test_dataloader = test_dataloader
        self.class_names = class_names
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Realiza avaliação completa do modelo.
        
        Returns:
            Dicionário com todas as métricas e caminhos de artefatos.
        """
        print("Iniciando avaliação do modelo...")
        
        y_true, y_pred, y_pred_proba = self._get_predictions()
        
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        report_str, report_path = self._generate_classification_report(y_true, y_pred)
        
        cm_path = self._create_confusion_matrix(y_true, y_pred)
        
        predictions_path = self.save_predictions(y_true, y_pred, y_pred_proba)
        
        results = {
            **metrics,
            'classification_report_str': report_str,
            'classification_report_path': report_path,
            'confusion_matrix_path': cm_path,
            'predictions_path': predictions_path,
            'num_samples': len(y_true),
        }
        
        print("✅ Avaliação concluída!")
        return results
    
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Obtém predições do modelo para o conjunto de teste."""
        self.model.eval()
        all_true, all_pred, all_proba = [], [], []
        
        print("Gerando predições...")
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                # Detecta se o batch tem features extras (3 tensores) ou só imagem (2 tensores)
                if len(batch) == 3:
                    data, feats, target = batch
                    data, feats, target = data.to(self.device), feats.to(self.device), target.to(self.device)
                    output = self.model(data, feats)
                else:
                    data, target = batch
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                
                proba = torch.softmax(output, dim=1)
                pred = torch.argmax(proba, dim=1)
                
                all_true.extend(target.cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                all_proba.extend(proba.cpu().numpy())
                
                if (i + 1) % 10 == 0:
                    print(f"  Processados {(i + 1) * self.test_dataloader.batch_size} amostras...")
        
        return np.array(all_true), np.array(all_pred), np.array(all_proba)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calcula um conjunto expandido de métricas de avaliação."""
        metrics = {}
        
        # Métricas gerais
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Métricas ponderadas e macro
        for avg in ['weighted', 'macro']:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=avg, zero_division=0
            )
            metrics[f'precision_{avg}'] = precision
            metrics[f'recall_{avg}'] = recall
            metrics[f'f1_{avg}'] = f1
            
        # Métricas por classe
        prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=np.arange(len(self.class_names)), zero_division=0
        )
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = prec_per_class[i]
            metrics[f'recall_{class_name}'] = rec_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]

        # AUC-ROC e Especificidade
        try:
            if len(self.class_names) == 2:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                metrics['auc_roc_ovr_weighted'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Aviso: Não foi possível calcular AUC-ROC/Especificidade: {e}")

        return metrics
    
    def _generate_classification_report(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray) -> Tuple[str, str]:
        """Gera e salva o relatório de classificação detalhado."""
        report_str = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )
        
        os.makedirs('artifacts', exist_ok=True)
        report_path = 'artifacts/classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("RELATÓRIO DE CLASSIFICAÇÃO\n")
            f.write("="*50 + "\n\n")
            f.write(report_str)
        
        print(f"📋 Relatório de classificação salvo em: {report_path}")
        return report_str, report_path
    
    def _create_confusion_matrix(self, y_true: np.ndarray, 
                               y_pred: np.ndarray) -> str:
        """Cria e salva a matriz de confusão."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Número de Amostras'}
        )
        
        plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
        plt.xlabel('Predições', fontsize=12)
        plt.ylabel('Labels Verdadeiros', fontsize=12)
        plt.tight_layout()
        
        os.makedirs('artifacts', exist_ok=True)
        cm_path = 'artifacts/confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Matriz de confusão salva em: {cm_path}")
        return cm_path
    
    def save_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_pred_proba: np.ndarray, 
                        output_path: str = 'artifacts/predictions.csv') -> str:
        """Salva predições detalhadas em arquivo CSV."""
        data = {
            'true_label': y_true,
            'predicted_label': y_pred,
            'true_class': [self.class_names[i] for i in y_true],
            'predicted_class': [self.class_names[i] for i in y_pred],
            'correct': y_true == y_pred
        }
        
        for i, class_name in enumerate(self.class_names):
            data[f'proba_{class_name}'] = y_pred_proba[:, i]
            
        df = pd.DataFrame(data)
        os.makedirs('artifacts', exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"💾 Predições salvas em: {output_path}")
        return output_path 