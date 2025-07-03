"""
Sistema de Deep Learning para Detecção de Autismo
"""

import argparse
import sys
import os
import yaml
from trainer import Trainer
from model_factory import ModelFactory
from models.model_registry import ModelRegistry
from data_handler import DataHandler


def load_config(config_path: str = 'config.yaml'):
    """Carrega configuração do arquivo YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_num_classes(config):
    """Detecta o número de classes do dataset."""
    data_handler = DataHandler(config['data'])
    dataset_info = data_handler.load_dataset()
    return dataset_info['num_classes']


def run_training(model_name: str = None, config_path: str = 'config.yaml'):
    """
    Executa treinamento de um modelo específico ou todos da lista.
    
    Args:
        model_name: Nome do modelo específico ou None para treinar todos
        config_path: Caminho para arquivo de configuração
    """
    config = load_config(config_path)
    trainer = Trainer(config_path)
    
    # Detectar número de classes
    num_classes = get_num_classes(config)
    print(f"Número de classes detectadas: {num_classes}")
    
    # Determinar quais modelos treinar
    if model_name:
        # Treinar modelo específico
        models_to_train = [model_name]
        print(f"Treinando modelo específico: {model_name}")
    else:
        # Treinar todos os modelos da lista no config
        models_to_train = config['models']['model_list']
        if not models_to_train:
            print("Nenhum modelo configurado em 'models.model_list'")
            return {}
        print(f"Treinando {len(models_to_train)} modelos da configuração")
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Treinando: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Criar modelo
            model = ModelFactory.create_model(
                model_name=model_name,
                input_shape=(config['data']['image_height'], 
                           config['data']['image_width'], 
                           config['data']['channels']),
                num_classes=num_classes,  # Dinâmico baseado no dataset
                config=ModelFactory.get_model_config(model_name)
            )
            
            # Mostrar configuração
            model_config = ModelFactory.get_model_config(model_name)
            print(f"Configuração do {model_name}:")
            print(f"  Épocas: {model_config.get('epochs', 'default')}")
            print(f"  Learning rate: {model_config.get('learning_rate', 'default')}")
            print(f"  Batch size: {model_config.get('batch_size', 'default')}")
            
            # Treinar modelo
            print(f"\nIniciando treinamento do {model_name}...")
            result = trainer.train_model(model, model_name)
            results[model_name] = result
            
            print(f"\nTreinamento concluído!")
            print(f"Acurácia de teste: {result.get('accuracy', 0):.4f}")
            print(f"F1-Score: {result.get('f1_weighted', 0):.4f}")
            
        except Exception as e:
            print(f"Erro ao treinar {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    return results


def show_models_info(config_path: str = 'config.yaml'):
    """Mostra informações sobre os modelos disponíveis."""
    config = load_config(config_path)
    
    print("INFORMAÇÕES DOS MODELOS")
    print("=" * 40)
    
    # Modelos disponíveis no registry
    available_models = ModelRegistry.get_available_models()
    print(f"\nModelos disponíveis no sistema: {len(available_models)}")
    for model in available_models:
        print(f"  - {model}")
    
    # Modelos configurados para treinar
    configured_models = config['models']['model_list']
    print(f"\nModelos configurados para treinar: {len(configured_models)}")
    for model in configured_models:
        print(f"  - {model}")
    
    # Configuração do dataset
    print(f"\nConfiguração do dataset:")
    print(f"  Caminho: {config['data']['dataset_path']}")
    print(f"  Tamanho das imagens: {config['data']['image_height']}x{config['data']['image_width']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    
    # Detectar número de classes
    try:
        num_classes = get_num_classes(config)
        print(f"  Número de classes: {num_classes}")
    except:
        print(f"  Número de classes: (dataset não encontrado)")
    
    # Configuração do MLflow
    print(f"\nConfiguração do MLflow:")
    print(f"  Experimento: {config['mlflow']['experiment_name']}")
    print(f"  Tracking URI: {config['mlflow']['tracking_uri']}")


def test_models(model_name: str = None, config_path: str = 'config.yaml'):
    """Testa se os modelos podem ser criados corretamente."""
    config = load_config(config_path)
    
    # Detectar número de classes
    try:
        num_classes = get_num_classes(config)
    except:
        print("Aviso: Dataset não encontrado, usando 2 classes para teste")
        num_classes = 2
    
    # Determinar quais modelos testar
    if model_name:
        models_to_test = [model_name]
    else:
        models_to_test = ModelRegistry.get_available_models()
    
    print("TESTE DOS MODELOS")
    print("=" * 30)
    
    for model_name in models_to_test:
        print(f"\nTestando {model_name}...")
        
        try:
            # Criar modelo
            model = ModelFactory.create_model(
                model_name=model_name,
                input_shape=(224, 224, 3),
                num_classes=num_classes,
                config=ModelFactory.get_model_config(model_name)
            )
            
            # Testar forward pass
            import torch
            model.eval()
            test_input = torch.randn(1, 3, 224, 224)
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"  OK - Output shape: {output.shape}")
            
        except Exception as e:
            print(f"  ERRO: {str(e)}")
    
    print("\nTeste concluído!")


def main():
    """Função principal com interface de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Sistema de Deep Learning para Detecção de Autismo"
    )
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'test', 'info', 'list'],
        default='info',
        help='Modo de execução (default: info)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Nome do modelo específico (opcional)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Caminho para arquivo de configuração (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Verificar se arquivo de configuração existe
    if not os.path.exists(args.config):
        print(f"Erro: Arquivo de configuração não encontrado: {args.config}")
        sys.exit(1)
    
    try:
        if args.mode == 'info':
            show_models_info(args.config)
            
        elif args.mode == 'list':
            # Listar modelos disponíveis
            print("Modelos disponíveis:")
            for model in ModelRegistry.get_available_models():
                print(f"  - {model}")
                
        elif args.mode == 'test':
            test_models(args.model, args.config)
            
        elif args.mode == 'train':
            results = run_training(args.model, args.config)
            
            # Resumo dos resultados
            if results:
                print("\n" + "="*60)
                print("RESUMO DOS RESULTADOS")
                print("="*60)
                for model_name, result in results.items():
                    if 'error' in result:
                        print(f"{model_name}: ERRO - {result['error']}")
                    else:
                        print(f"{model_name}: Acurácia={result.get('accuracy', 0):.4f}")
            
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nErro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Se não houver argumentos, mostrar ajuda
    if len(sys.argv) == 1:
        print("SISTEMA DE DETECÇÃO DE AUTISMO")
        print("=" * 40)
        print("\nUso:")
        print("  python main.py --mode train              # Treinar todos os modelos do config")
        print("  python main.py --mode train --model CNN  # Treinar modelo específico")
        print("  python main.py --mode test               # Testar todos os modelos")
        print("  python main.py --mode info               # Informações do sistema")
        print("  python main.py --mode list               # Listar modelos disponíveis")
        print("\nPara mais opções: python main.py --help")
        print()
    
    main() 