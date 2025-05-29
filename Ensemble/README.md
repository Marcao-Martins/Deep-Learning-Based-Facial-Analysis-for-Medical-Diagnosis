# Ensemble CNN Classification for Autism Detection

This directory contains an implementation for classifying autism using an ensemble of pre-trained Convolutional Neural Networks (CNNs): EfficientNetB5, MobileNetV2, and InceptionV3. The approach is based on the paper "Prediction and Evaluation of Autism Spectrum Disorder using AI-enabled Convolutional Neural Network and Transfer Learning : An Ensemble Approach."

It utilizes the `my_mlflow_utils` framework for MLflow integration and experiment tracking.

## Features

-   Fine-tunes three pre-trained models: EfficientNetB5, MobileNetV2, and InceptionV3.
-   Applies data augmentation techniques (rotation, shifts, shearing, zooming, horizontal flipping) during training.
-   Uses soft voting to combine predictions from the individual models for an ensemble prediction.
-   Logs individual model training progress (epoch-wise metrics) to MLflow.
-   Logs individual model artifacts (trained weights) to MLflow.
-   Logs detailed evaluation metrics for each individual model (accuracy, precision, recall, F1, ROC AUC, confusion matrix) to MLflow.
-   Logs overall ensemble performance metrics (accuracy, precision, recall, F1, ROC AUC, confusion matrix) to MLflow.

## Setup

1.  **Clone the Repository (if you haven't already):**
    ```bash
    # git clone ...
    cd path/to/Yolo and Detectron
    ```

2.  **Install Dependencies:**
    Navigate to the `Ensemble` directory and install the required Python packages:
    ```bash
    cd Ensemble
    pip install -r requirements.txt
    ```
    Ensure `my_mlflow_utils` is accessible (the `train_ensemble.py` script attempts to add the parent directory to `sys.path`).

3.  **Dataset:**
    Ensure the Autism dataset is available and organized as expected by `AutismDataLoader` (typically in a directory like `../AutismDataset` relative to the `Ensemble` directory, or specify the path using `--dataset`):
    ```
    Parent_Directory_Of_Dataset/
    └── AutismDataset/
        ├── train/
        │   ├── Autistic/       (images)
        │   └── Non_Autistic/   (images)
        ├── valid/
        │   ├── Autistic/       (images)
        │   └── Non_Autistic/   (images)
        └── test/
            ├── Autistic/       (images)
            └── Non_Autistic/   (images)
    ```

## Usage

To train and evaluate the ensemble model with default settings:

```bash
python train_ensemble.py
```

This will train EfficientNetB5, MobileNetV2, and InceptionV3 sequentially, log their individual performances, and then evaluate the ensemble using soft voting.

### Command-line Options

-   `--dataset`: Path to the Autism dataset directory (default: `../AutismDataset`).
-   `--experiment`: Name of the MLflow experiment (default: `Autism_Ensemble_Classification`).
-   `--run-name`: Name for this specific MLflow run (default: `Ensemble_EffNetB5_MobileNetV2_InceptionV3`).
-   `--epochs`: General number of training epochs for each model (default: 10).
-   `--lr`: General learning rate for each model (default: 0.001).
-   `--weight-decay`: General weight decay for the optimizer for each model (default: 1e-5).
-   **Model-Specific Parameters:** You can also provide model-specific parameters by prefixing them with the model name (e.g., `--EfficientNetB5_epochs 15`, `--MobileNetV2_lr 0.0005`). The script `train_ensemble.py` has examples for `EfficientNetB5_epochs` and `MobileNetV2_lr`.

### Examples

-   Train with default parameters:
    ```bash
    python train_ensemble.py
    ```

-   Specify dataset path and number of epochs for all models:
    ```bash
    python train_ensemble.py --dataset /path/to/your/AutismDataset --epochs 15
    ```

-   Specify different epochs for EfficientNetB5:
    ```bash
    python train_ensemble.py --EfficientNetB5_epochs 20 --epochs 10 
    # (EfficientNetB5 will train for 20 epochs, others for 10)
    ```