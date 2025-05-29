# EfficientNet Autism Classification

This directory contains an implementation of an EfficientNet model for autism classification using the MLflow utilities framework.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the Autism dataset is organized with the following structure:
   - AutismDataset/
     - train/
       - Autistic/
       - Non_Autistic/
     - valid/
       - Autistic/
       - Non_Autistic/
     - test/
       - Autistic/
       - Non_Autistic/

## Usage

To train the model with default settings:

```bash
python train.py
```

### Command-line Options

- `--dataset`: Path to the Autism dataset directory (default: "../AutismDataset")
- `--experiment`: Name of the MLflow experiment (default: "Autism_Classification")
- `--run-name`: Name of the specific run (default: "EfficientNet_B0")
- `--model`: EfficientNet model variant (default: "efficientnet_b0", choices: b0-b7)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay for optimizer (default: 1e-5)

### Examples

Train with EfficientNet-B3 for 20 epochs:

```bash
python train.py --model efficientnet_b3 --epochs 20
```

Use a different learning rate:

```bash
python train.py --lr 0.0005
```

## Viewing Results

After training, you can view the results using MLflow:

```bash
mlflow ui
```

Then open your browser at http://127.0.0.1:5000 to access the MLflow UI. 