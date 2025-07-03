"""
Factory para criação de modelos usando o sistema modular com PyTorch.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from models.model_registry import ModelRegistry


class ModelFactory:
    """Factory para criação de modelos."""
    
    @staticmethod
    def create_model(model_name: str, input_shape: Tuple[int, int, int] = (224, 224, 3), 
                    num_classes: int = 2, config: Dict[str, Any] = None) -> nn.Module:
        """
        Cria um modelo baseado no nome.
        
        Args:
            model_name: Nome do modelo
            input_shape: Formato da entrada (height, width, channels)
            num_classes: Número de classes
            config: Configurações específicas do modelo (opcional)
            
        Returns:
            Modelo PyTorch (nn.Module)
        """
        # Obter classe do modelo do registro
        model_class = ModelRegistry.get_model(model_name)
        
        # Criar instância do modelo
        model_instance = model_class()
        
        # Construir modelo
        model = model_instance.build(input_shape, num_classes, config)
        
        return model
    
    @staticmethod
    def get_available_models() -> list:
        """
        Retorna lista de modelos disponíveis.
        
        Returns:
            Lista com nomes dos modelos
        """
        return ModelRegistry.get_available_models()
    
    @staticmethod
    def get_model_config(model_name: str) -> dict:
        """
        Obtém configuração padrão de um modelo.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Configuração padrão do modelo
        """
        return ModelRegistry.get_model_config(model_name)
    
    @staticmethod
    def get_all_configs() -> Dict[str, dict]:
        """
        Obtém configurações de todos os modelos.
        
        Returns:
            Dicionário com configurações de todos os modelos
        """
        return ModelRegistry.get_all_configs() 