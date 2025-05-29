# ResNet-50 Pipeline for Autism Classification

This directory contains the implementation of a ResNet-50 based image classification pipeline for the Autism dataset, inspired by the paper "Efficient Classification Of Autism in Children based on Resnet-50 and Xception module."

It leverages the shared `my_mlflow_utils` for experiment tracking, data loading, and standardized metric logging with MLflow.

## Structure

- `resnet_pipeline.py`: Defines the `ResNetPipeline` class, which handles:
    - Data loading and preprocessing (using `AutismImageDataset`).
    - ResNet-50 model creation with transfer learning (freezing base layers, replacing the final classifier).
    - Model training (optimizing only the new classifier head).
    - Model evaluation using the standardized `log_standard_classification_metrics` from `my_mlflow_utils` (prefix: `eval`).
- `train_resnet.py`: The main script to run the ResNet-50 pipeline. It parses command-line arguments for dataset paths, MLflow configuration, and hyperparameters.
- `requirements.txt`: Lists the necessary Python dependencies.
- `README.md`: This file.

## Prerequisites

1.  **Python Environment**: Ensure you have Python 3.8+ installed.
2.  **Dependencies**: Install the required packages. It's recommended to use a virtual environment:
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    # source .venv/bin/activate
    pip install -r requirements.txt
    pip install -r ../my_mlflow_utils/requirements.txt # If my_mlflow_utils has its own, or ensure all are in a shared one.
    ```
3.  **Dataset**: Prepare your Autism image dataset. It should have a root directory with subdirectories for each split (`train`, `valid`, `test`), and within each split, subdirectories for each class (`Autistic`, `Non_Autistic`). Example structure:
    ```
    <dataset_path>/
    ├── train/
    │   ├── Autistic/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── Non_Autistic/
    │       ├── img1.jpg
    │       └── ...
    ├── valid/
    │   ├── Autistic/
    │   └── Non_Autistic/
    └── test/
        ├── Autistic/
        └── Non_Autistic/
    ```
4.  **MLflow**: Ensure MLflow is installed (`pip install mlflow`). By default, runs will be logged to a local `mlruns` directory (this is configured in `my_mlflow_utils/base_pipeline.py` to be at the project root).

## How to Run

Navigate to the `ResNet` directory in your terminal.

```bash
python train_resnet.py --dataset_path /path/to/your/autism_dataset \
                       --experiment_name "Autism_ResNet_Experiments" \
                       --run_name "ResNet50_TransferLearning_Run1" \
                       --model_name_tag "ResNet50_Autism" \
                       --epochs 25 \
                       --learning_rate 0.001 \
                       --weight_decay 0.0001
```

### Command-Line Arguments for `train_resnet.py`:

- `--dataset_path` (required): Path to the root of your Autism image dataset.
- `--experiment_name` (optional): Name for the MLflow experiment. Default: `"Autism_Classification_ResNet"`.
- `--run_name` (optional): Name for this specific MLflow run. Default: `"ResNet50_Run"`.
- `--model_name_tag` (optional): A tag to identify the model family in MLflow. Default: `"ResNet50"`.
- `--epochs` (optional): Number of training epochs. Default: `25`.
- `--learning_rate` (optional): Learning rate for the optimizer. Default: `0.001`.
- `--weight_decay` (optional): Weight decay for the optimizer. Default: `0.0001`.
- `--num_classes` (optional): Number of classes. Default: `2`.

## Viewing Results in MLflow

1.  Navigate to the parent directory of your `mlruns` folder (likely your project root).
2.  Run the MLflow UI:
    ```bash
    mlflow ui
    ```
3.  Open your browser to `http://localhost:5000` (or the address shown in the terminal).
    You should see your experiment (`Autism_ResNet_Experiments` or your custom name) and the runs within it, with parameters and standardized `eval_...` metrics. 