# DeepVision: Flexible Medical Image Classification Framework

A PyTorch Lightning-based framework for training and evaluating various deep learning models on medical image datasets. The framework is designed to be both model-agnostic and dataset-agnostic, allowing easy experimentation with different architectures and datasets.

## Project Overview

This project implements a flexible deep learning pipeline that can work with:
- Any image classification model from HuggingFace
- Any image dataset through custom DataModules
- Currently configured for DermaMNIST as an example

## Model Support

The framework supports a wide range of model architectures from HuggingFace, including but not limited to:

- Vision Transformers (ViT)
- Swin Transformers
- DeiT (Data-efficient Image Transformers)
- ConvNeXT
- BEiT

## Dataset Support

The framework can be easily adapted to work with different datasets:

1. Built-in support for:
   - Medical image datasets (e.g., DermaMNIST)
   - Standard vision datasets (e.g., ImageNet-style)
   - Custom datasets

2. Dataset requirements:
   - Image data in any common format
   - Classification labels
   - Train/validation/test splits (can be created automatically)

## Project Structure

```
deeplearning_ViT_PCAM/
├── src/
│   ├── data/              # Dataset modules
│   │   └── dermmnist_datamodule.py
│   ├── models/           # Model implementations
│   │   └── model_factory.py
│   └── utils/            # Utility functions
│       └── metrics.py
├── configs/              # Configuration files
│   └── config.yaml
├── data/                 # Dataset storage
├── logs/                 # Training logs
├── results/              # Evaluation results
├── benchmark.py          # Main training script
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Dataset Configuration

To use a different dataset:

1. Create a new DataModule:
```python
# src/data/your_dataset_module.py
class YourDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, image_size=224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # Define your transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[...], std=[...])
        ])

    def setup(self, stage=None):
        # Load your dataset
        self.train_dataset = YourDataset(...)
        self.val_dataset = YourDataset(...)
        self.test_dataset = YourDataset(...)
```

2. Update config.yaml:
```yaml
dataset:
  data_dir: "data/your_dataset"
  batch_size: 32
  num_workers: 4
  image_size: 224
  mean: [0.485, 0.456, 0.406]  # Update with your dataset's values
  std: [0.229, 0.224, 0.225]   # Update with your dataset's values
```

Example configurations for different datasets:

```yaml
# ImageNet-style dataset
dataset:
  data_dir: "data/imagenet"
  batch_size: 32
  num_workers: 4
  image_size: 224
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

# CIFAR-style dataset
dataset:
  data_dir: "data/cifar"
  batch_size: 128
  num_workers: 4
  image_size: 224
  mean: [0.4914, 0.4822, 0.4465]
  std: [0.2023, 0.1994, 0.2010]

# Custom medical dataset
dataset:
  data_dir: "data/medical"
  batch_size: 16
  num_workers: 4
  image_size: 224
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]
```

## Model Configuration

To use a different model, update the config.yaml:

```yaml
models:
  - name: "your-chosen-model"  # HuggingFace model identifier
    num_classes: N             # Number of classes in your dataset
    learning_rate: 0.0001
    weight_decay: 0.01
```

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
training:
  max_epochs: 10
  accelerator: "gpu"
  devices: 1
  strategy: "auto"
  precision: 32

logging:
  project_name: "your-project-name"
  save_dir: "logs"
  log_every_n_steps: 50
```

## Usage

Run the training script:
```bash
python benchmark.py
```

The script will:
1. Load the specified dataset
2. Initialize the chosen model
3. Train using the specified configuration
4. Log metrics and save checkpoints

## Dataset Tips

When adapting to a new dataset:

1. Data Preparation:
   - Ensure consistent image formats
   - Create proper train/val/test splits
   - Calculate dataset statistics (mean/std)

2. Performance Optimization:
   - Adjust batch size based on image size and memory
   - Tune number of workers for data loading
   - Consider adding dataset-specific augmentations

3. Common Issues:
   - Memory management for large datasets
   - Handling imbalanced classes
   - Dealing with missing or corrupt data

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
- torchvision
- Weights & Biases
- scikit-learn

For a complete list of dependencies, see `requirements.txt`.

## License

This project is licensed under the MIT License.
