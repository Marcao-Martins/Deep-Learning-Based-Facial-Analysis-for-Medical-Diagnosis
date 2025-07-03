# Sistema de Deep Learning Modular

Um sistema modular e escalável para experimentar com diferentes arquiteturas de deep learning em tarefas de classificação de imagens, com tracking completo de experimentos usando MLflow.

## 🏗️ Arquitetura Modular

O projeto foi redesenhado com uma arquitetura altamente modular onde cada modelo tem seu próprio arquivo e configurações:

```
models/
├── __init__.py
├── base_model.py           # Classe base abstrata
├── model_registry.py       # Registro central de modelos
├── simple_vgg.py          # Modelo VGG simplificado
├── resnet50_model.py      # ResNet50 com transfer learning
├── mobilenetv2_model.py   # MobileNetV2 com transfer learning
├── densenet121_model.py   # DenseNet121 com transfer learning
├── efficientnetb0_model.py # EfficientNetB0 com transfer learning
├── inceptionv3_model.py   # InceptionV3 com transfer learning
└── cnn_custom.py          # Arquitetura CNN customizada
```

## ✨ Características Principais

### 🔧 Modularidade Extrema
- **Arquivos separados para cada modelo**: Cada arquitetura tem seu próprio arquivo Python
- **Configurações embutidas**: Cada modelo define suas próprias configurações padrão
- **Registro central**: Sistema de registro para gerenciar todos os modelos disponíveis
- **Fácil extensão**: Adicionar novos modelos é simples e não requer mudanças no código principal

### 📊 Tracking Completo com MLflow
- **Parâmetros**: Todos os hiperparâmetros são automaticamente registrados
- **Métricas**: Acompanhamento de accuracy, F1-score, precision, recall, etc.
- **Artefatos**: Modelos salvos, gráficos de confusão, relatórios de avaliação
- **Experimentos organizados**: Cada modelo gera um experimento separado

### 🎯 Configurações Flexíveis
- **Por modelo**: Cada modelo tem suas próprias configurações otimizadas
- **Estratégias de treinamento**: Freeze/unfreeze, learning rates, batch sizes específicos
- **Data augmentation**: Configurável por modelo
- **Callbacks personalizados**: Early stopping, learning rate reduction, checkpointing

## 🚀 Modelos Disponíveis

### 1. **SimpleVGG** (`simple_vgg.py`)
- Arquitetura VGG simplificada
- Configurações: 100 épocas, lr=0.001, batch_size=32
- Ideal para: Baseline e comparação

### 2. **ResNet50** (`resnet50_model.py`)
- Transfer learning com ResNet50 pré-treinado
- Configurações: 150 épocas, lr=0.0001, batch_size=16
- Estratégia: Freeze inicial + unfreeze na época 50

### 3. **MobileNetV2** (`mobilenetv2_model.py`)
- Modelo leve e eficiente
- Configurações: 200 épocas, lr=0.0001, batch_size=32
- Estratégia: Freeze inicial + unfreeze na época 100

### 4. **DenseNet121** (`densenet121_model.py`)
- Arquitetura densa com conexões diretas
- Configurações: 120 épocas, lr=0.0001, batch_size=16
- Estratégia: Freeze inicial + unfreeze na época 60

### 5. **EfficientNetB0** (`efficientnetb0_model.py`)
- Modelo eficiente com scaling automático
- Configurações: 180 épocas, lr=0.0001, batch_size=16
- Estratégia: Freeze inicial + unfreeze na época 90

### 6. **InceptionV3** (`inceptionv3_model.py`)
- Arquitetura com múltiplas escalas
- Configurações: 250 épocas, lr=0.0001, batch_size=8
- Estratégia: Freeze inicial + unfreeze na época 125

### 7. **CNN_Custom** (`cnn_custom.py`)
- Arquitetura customizada complexa
- Configurações: 2000 épocas, lr=0.001, batch_size=32
- Recursos: Batch normalization, global pooling configurável

## 📁 Estrutura do Projeto

```
├── models/                     # Pacote de modelos
│   ├── __init__.py
│   ├── base_model.py          # Classe base abstrata
│   ├── model_registry.py      # Registro central
│   ├── resnet50_model.py      # ResNet50
│   ├── mobilenetv2_model.py   # MobileNetV2
│   ├── densenet121_model.py   # DenseNet121
│   ├── efficientnetb0_model.py # EfficientNetB0
│   ├── inceptionv3_model.py   # InceptionV3
├── data/                      # Diretório de dados
├── config.yaml               # Configuração geral
├── data_handler.py           # Gerenciamento de dados
├── model_factory.py          # Factory de modelos
├── trainer.py                # Sistema de treinamento
├── evaluator.py              # Avaliação de modelos
├── main.py                   # Script principal
├── inference.py              # Inferência
└── requirements.txt          # Dependências
```

## ⚙️ Configuração

### 1. Instalação das Dependências
```bash
pip install -r requirements.txt
```

### 2. Configuração do Dataset
Organize seu dataset na estrutura:
```
data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── valid/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

### 3. Configuração Geral (`config.yaml`)
```yaml
# MLflow Configuration
mlflow:
  experiment_name: "autism_detection_experiments"
  tracking_uri: "file:./mlruns"

# Data Configuration
data:
  dataset_path: "data"
  image_height: 224
  image_width: 224
  channels: 3
  batch_size: 32
  validation_split: 0.15
  test_split: 0.15

# Training Configuration (padrão - será sobrescrito por configurações específicas dos modelos)
training:
  epochs: 50
  initial_learning_rate: 0.001
  optimizer: "adam"
  loss: "categorical_crossentropy"
  metrics: ["accuracy"]

# Model Configuration
models:
  model_list: []  # Adicione modelos conforme necessário
```

## 🚀 Uso

### Execução Completa
```bash
python main.py
```

### Execução de Modelo Específico
```python
from trainer import ExperimentTrainer
from model_factory import ModelFactory

# Carregar configuração
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Inicializar trainer
trainer = ExperimentTrainer(config)

# Treinar modelo específico
result = trainer.train_model("ResNet50")
```

### Adicionando um Novo Modelo

1. **Criar arquivo do modelo** (`models/meu_modelo.py`):
```python
from .base_model import BaseModel
import tensorflow as tf

class MeuModelo(BaseModel):
    @classmethod
    def get_model_name(cls) -> str:
        return "MeuModelo"
    
    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "epochs": 100,
            "initial_learning_rate": 0.001,
            "batch_size": 32,
            # ... outras configurações
        }
    
    def build(self, input_shape, num_classes, config=None):
        # Implementar arquitetura do modelo
        pass
```

2. **Registrar no registry** (`models/model_registry.py`):
```python
from .meu_modelo import MeuModelo

_models = {
    # ... modelos existentes
    'MeuModelo': MeuModelo
}
```

3. **Adicionar à lista de modelos** (`config.yaml`):
```yaml
models:
  model_list:
    - "MeuModelo"
```

## 📊 Visualização de Resultados

### MLflow UI
```bash
mlflow ui
```
Acesse: http://localhost:5000

**Recursos disponíveis no MLflow:**
- 📈 Gráficos de métricas de treinamento
- 📊 Comparação de modelos lado a lado
- 🎯 Análise detalhada de parâmetros
- 💾 Download de modelos treinados
- 📋 Relatórios de avaliação
- 🔍 Filtros e ordenação por métricas

## 🔍 Funcionalidades Avançadas

### Configurações Customizadas por Modelo
Cada modelo define suas próprias configurações:
- **Épocas**: Diferentes durações de treinamento
- **Learning rates**: Otimizados para cada arquitetura
- **Batch sizes**: Ajustados para memória disponível
- **Estratégias de freeze/unfreeze**: Para transfer learning
- **Data augmentation**: Configurável por modelo

### Callbacks Inteligentes
- **Early Stopping**: Patience configurável por modelo
- **Learning Rate Reduction**: Redução automática na plateau
- **Model Checkpointing**: Salvamento do melhor modelo
- **Unfreeze Callback**: Descongelamento automático de camadas

### Avaliação Abrangente
- **Métricas**: Accuracy, Precision, Recall, F1-score
- **Matriz de Confusão**: Visualização detalhada
- **Relatórios**: Classificação completa
- **Artefatos**: Gráficos e relatórios salvos


Para adicionar novos modelos:

1. Crie um novo arquivo em `models/`
2. Implemente a classe herdando de `BaseModel`
3. Registre no `ModelRegistry`
4. Adicione à lista de modelos no `config.yaml`

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.
