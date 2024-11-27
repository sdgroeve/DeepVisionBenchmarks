# DeepVision: Flexible Medical Image Classification Framework

A PyTorch Lightning-based framework for training and evaluating various deep learning models on the DermaMNIST dataset. The framework is designed to be model-agnostic, allowing easy experimentation with different architectures from HuggingFace's model hub.

## Project Overview

This project implements a flexible deep learning pipeline for medical image classification that can work with any image classification model from HuggingFace. It currently uses the DermaMNIST dataset, which contains dermatoscopic images for skin lesion classification.

## Model Support

The framework supports a wide range of model architectures from HuggingFace, including but not limited to:

- Vision Transformers (ViT) - various sizes and configurations
- Swin Transformers
- DeiT (Data-efficient Image Transformers)
- ConvNeXT
- BEiT (Bidirectional Encoder representation from Image Transformers)

## Project Structure

```
DeepVisionBenchmarks/
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

## Model Configuration

The framework is designed to work with any image classification model from HuggingFace. To use a different model:

1. Visit [HuggingFace Model Hub](https://huggingface.co/models?pipeline_tag=image-classification)
2. Choose any model that supports image classification
3. Update the `configs/config.yaml` file:

```yaml
models:
  - name: "your-chosen-model"  # HuggingFace model identifier
    num_classes: 7             # Number of classes (7 for DermMNIST)
    learning_rate: 0.0001      # Adjust based on model size
    weight_decay: 0.01         # Adjust based on training behavior
```

Example configurations for different models:

```yaml
# Vision Transformer (ViT)
- name: "google/vit-base-patch16-224"
  num_classes: 7
  learning_rate: 0.0001
  weight_decay: 0.01

# Swin Transformer
- name: "microsoft/swin-tiny-patch4-window7-224"
  num_classes: 7
  learning_rate: 0.0001
  weight_decay: 0.01

# DeiT
- name: "facebook/deit-base-patch16-224"
  num_classes: 7
  learning_rate: 0.0001
  weight_decay: 0.01

# ConvNeXT
- name: "facebook/convnext-tiny-224"
  num_classes: 7
  learning_rate: 0.0001
  weight_decay: 0.01
```

## Dataset

The project uses the DermaMNIST dataset from the MedMNIST collection. Key features:
- Image size: Automatically resized to match model requirements
- Channels: Converted to 3-channel RGB
- Normalization: Mean [0.5, 0.5, 0.5], Std [0.5, 0.5, 0.5]
- Data splits: Train, Validation, and Test sets

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

## Training Configuration

The training configuration in `configs/config.yaml` includes:

```yaml
dataset:
  data_dir: "data/dermmnist"
  batch_size: 32
  num_workers: 4
  image_size: 224  # Adjust based on model requirements

training:
  max_epochs: 10
  accelerator: "gpu"
  devices: 1
  strategy: "auto"
  precision: 32  # Use 16 for faster training if supported

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
2. Initialize the specified model from HuggingFace
3. Train the model using the specified configuration
4. Log metrics and save checkpoints

## Model Selection Tips

When choosing a model:
1. Consider the model size vs. your computational resources
2. Check the required input image size
3. Adjust learning rate based on model size:
   - Smaller models (tiny/small): 1e-4 to 5e-4
   - Larger models (base/large): 1e-5 to 1e-4
4. Adjust weight decay based on training behavior:
   - Increase if overfitting (e.g., 0.1)
   - Decrease if underfitting (e.g., 0.001)

## Logging

Training progress and results are logged using Weights & Biases (wandb). Logs include:
- Training and validation metrics
- Model checkpoints
- Learning curves
- System metrics

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
