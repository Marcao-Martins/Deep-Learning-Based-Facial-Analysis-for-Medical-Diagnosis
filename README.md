# Autism Detection MLflow Framework

A robust, object-oriented framework for running and tracking machine learning experiments for Autism Spectrum Disorder (ASD) detection using facial images.

## Project Structure

```
.
├── AutismDataset/          # Your dataset directory
│   ├── train/
│   │   ├── Autistic/
│   │   └── Non_Autistic/
│   ├── valid/
│   │   ├── Autistic/
│   │   └── Non_Autistic/
│   └── test/
│       ├── Autistic/
│       └── Non_Autistic/
├── config.py               # Central configuration file
├── base_experiment.py      # Base class for all experiments
├── data_utils.py          # Data loading utilities
├── metrics_utils.py       # Metrics calculation and visualization
├── example_efficientnet_experiment.py  # Example implementation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run the Example EfficientNet Experiment

```bash
python example_efficientnet_experiment.py
```

### 2. View Results in MLflow UI

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Creating New Experiments

To create a new experiment for a different model, create a new Python file that inherits from `BaseExperiment`:

```python
from base_experiment import BaseExperiment
import data_utils
import metrics_utils
import config

class YourModelExperiment(BaseExperiment):
    def __init__(self, your_params):
        super().__init__()
        # Initialize your parameters
        
    def prepare_data(self):
        # Load and prepare your data
        # Use utilities from data_utils.py
        
    def create_model(self):
        # Define your model architecture
        # Log model parameters using self.log_params()
        
    def train(self):
        # Train your model
        # Log metrics using self.log_metrics()
        
    def evaluate(self):
        # Evaluate on test set
        # Use utilities from metrics_utils.py
```

## Framework Features

### Base Experiment Class
- Automatic MLflow tracking setup
- Standardized experiment pipeline
- Built-in error handling and logging
- Dataset information logging

### Data Utilities
- Support for TensorFlow and PyTorch data loading
- Built-in data augmentation options
- Flexible image preprocessing
- Support for different batch sizes and image sizes

### Metrics Utilities
- Comprehensive classification metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrix visualization
- ROC curves (for binary classification)
- Precision-Recall curves
- Training history plots
- Classification report heatmaps

### MLflow Integration
- Automatic parameter logging
- Step-wise metric tracking
- Model versioning
- Artifact storage (plots, model files, JSON data)
- Experiment comparison

## Running Multiple Experiments

Create a script to run multiple experiments with different parameters:

```python
from example_efficientnet_experiment import EfficientNetExperiment

# Define parameter combinations
experiments = [
    {"model_variant": "B0", "learning_rate": 0.001, "batch_size": 32},
    {"model_variant": "B0", "learning_rate": 0.0001, "batch_size": 16},
    {"model_variant": "B0", "learning_rate": 0.001, "batch_size": 32, "augment": False},
]

# Run all experiments
for params in experiments:
    experiment = EfficientNetExperiment(**params)
    experiment.run()
```

## Creating Ensemble Models

For ensemble experiments, you can create a new experiment class that loads multiple pre-trained models:

```python
class EnsembleExperiment(BaseExperiment):
    def __init__(self, model_uris):
        super().__init__()
        self.model_uris = model_uris  # MLflow model URIs
        
    def create_model(self):
        # Load multiple models using mlflow.load_model()
        self.models = []
        for uri in self.model_uris:
            model = mlflow.tensorflow.load_model(uri)
            self.models.append(model)
```

## Tips for Organization

1. **Folder Structure for Trials**: Create separate folders for different experiment types:
   ```
   experiments/
   ├── standalone_models/
   │   ├── efficientnet_experiments.py
   │   ├── resnet_experiments.py
   │   └── vgg_experiments.py
   ├── yolo_pipelines/
   │   ├── yolo_efficientnet_pipeline.py
   │   └── yolo_resnet_pipeline.py
   └── ensemble_models/
       └── ensemble_experiments.py
   ```

2. **Naming Conventions**: Use descriptive run names that include key parameters:
   ```python
   run_name = f"EfficientNetB0_lr{learning_rate}_bs{batch_size}_aug{augment}"
   ```

3. **Parameter Sweeps**: Use configuration files or parameter grids:
   ```python
   param_grid = {
       'learning_rate': [0.001, 0.0001],
       'batch_size': [16, 32],
       'model_variant': ['B0', 'B3']
   }
   ```

## Comparing Results

1. In MLflow UI:
   - Sort by metrics (e.g., test_accuracy)
   - Filter by parameters
   - Use the Compare feature for detailed analysis
   - Download artifacts for further analysis

2. Programmatically:
   ```python
   import mlflow
   
   # Get all runs from an experiment
   experiment = mlflow.get_experiment_by_name("ASD_Detection_Multi_Model")
   runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
   
   # Find best run by accuracy
   best_run = runs.loc[runs['metrics.test_accuracy'].idxmax()]
   ```

## Next Steps

1. Implement additional model architectures (ResNet, VGG, MobileNet)
2. Create YOLO-based detection pipelines
3. Implement ensemble methods
4. Add cross-validation support
5. Create automated hyperparameter tuning scripts
6. Build a custom dashboard using MLflow's API

## Troubleshooting

- If MLflow UI doesn't show experiments, check that `MLFLOW_TRACKING_URI` in `config.py` points to the correct location
- For GPU memory issues, reduce batch size or image size
- For slow training, ensure GPU is being used and consider reducing model complexity 