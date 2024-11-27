# DeepVision: Medical Image Classification with Vision Transformers

A PyTorch Lightning-based framework for training and evaluating Vision Transformer (ViT) models on the DermaMNIST dataset for medical image classification.

## Project Overview

This project implements a deep learning pipeline for medical image classification using state-of-the-art Vision Transformer models from HuggingFace. It focuses on the DermaMNIST dataset, which contains dermatoscopic images for skin lesion classification.

## Project Structure

```
deeplearning_ViT_PCAM/
├── src/
│   ├── data/              # Dataset handling modules
│   │   └── dermmnist_datamodule.py
│   ├── models/           # Model implementations
│   │   └── model_factory.py
│   └── utils/            # Utility functions
│       └── metrics.py
├── configs/              # Configuration files
│   └── config.yaml
├── data/                 # Dataset storage
│   └── dermmnist/
├── logs/                 # Training logs and checkpoints
├── results/              # Evaluation results
├── benchmark.py          # Main training script
├── requirements.txt      # Project dependencies
└── README.md            # Documentation
```

## Dataset

The project uses the DermaMNIST dataset, which is part of the MedMNIST collection. It consists of dermatoscopic images for skin lesion classification with 7 different categories. The dataset is automatically downloaded and preprocessed during training.

Key features:
- Image size: Resized to 224x224 pixels (to match ViT requirements)
- Channels: Converted to 3-channel RGB
- Normalization: Mean [0.5, 0.5, 0.5], Std [0.5, 0.5, 0.5]
- Data splits: Train, Validation, and Test sets

## Model Architecture

The project currently implements:
- Vision Transformer (ViT) base model with patch size 16x16
- Pretrained weights from "google/vit-base-patch16-224"
- Adapted for 7-class classification (DermaMNIST categories)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The training configuration is defined in `configs/config.yaml`:

```yaml
dataset:
  data_dir: "data/dermmnist"
  batch_size: 32
  num_workers: 4
  image_size: 224

models:
  - name: "google/vit-base-patch16-224"
    num_classes: 7
    learning_rate: 0.0001
    weight_decay: 0.01

training:
  max_epochs: 10
  accelerator: "gpu"
  devices: 1
  strategy: "auto"
  precision: 32

logging:
  project_name: "dermmnist-benchmark"
  save_dir: "logs"
  log_every_n_steps: 50
```

## Usage

Run the training script:
```bash
python benchmark.py
```

The script will:
1. Download and prepare the DermaMNIST dataset
2. Initialize the ViT model
3. Train the model using the specified configuration
4. Log metrics and save checkpoints

## Logging

Training progress and results are logged using Weights & Biases (wandb). Logs include:
- Training and validation metrics
- Model checkpoints
- Learning curves
- System metrics

Logs and checkpoints are saved in the `logs/` directory.

## Requirements

Key dependencies:
- PyTorch Lightning
- Transformers (HuggingFace)
- MedMNIST
- Weights & Biases
- torchvision
- scikit-learn

For a complete list of dependencies, see `requirements.txt`.

## License

This project is licensed under the MIT License.
