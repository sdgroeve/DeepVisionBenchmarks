# DeepVision: Flexible Vision Model Training Framework

A PyTorch Lightning-based framework for training and evaluating various deep learning models on image classification tasks. The framework supports both HuggingFace datasets and custom local datasets.

## Project Overview

This project implements a flexible deep learning pipeline that can work with:
- Any image classification model from HuggingFace
- Any dataset from HuggingFace's datasets hub
- Local custom datasets through custom DataModules

## Dataset Support

### 1. HuggingFace Datasets
Directly load and use any dataset from HuggingFace's hub:

```yaml
# configs/config.yaml
dataset:
  type: "huggingface"
  name: "imagenet-1k"  # or any other dataset from HuggingFace
  batch_size: 32
  num_workers: 4
  image_size: 224
  image_key: "image"
  label_key: "label"
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

Popular HuggingFace datasets:
- `imagenet-1k`: Full ImageNet dataset
- `cifar10`: CIFAR-10 dataset
- `mnist`: MNIST dataset
- `fashion_mnist`: Fashion-MNIST dataset

### 2. Local Datasets
Use custom local datasets like DermMNIST:

```yaml
dataset:
  type: "dermmnist"
  data_dir: "data/dermmnist"
  batch_size: 32
  num_workers: 4
  image_size: 224
```

## Model Support

The framework supports various model architectures from HuggingFace:

```yaml
models:
  - name: "google/vit-base-patch16-224"
    num_classes: 1000  # Adjust based on dataset
    learning_rate: 0.0001
    weight_decay: 0.01
```

## Project Structure

```
deeplearning_ViT_PCAM/
├── src/
│   ├── data/              
│   │   ├── dermmnist_datamodule.py
│   │   └── huggingface_datamodule.py
│   ├── models/           
│   │   └── model_factory.py
│   └── utils/            
│       └── metrics.py
├── configs/              
│   └── config.yaml
├── data/                 
├── logs/                 
├── results/              
├── benchmark.py          
├── requirements.txt      
└── README.md            
```

## Dataset Configuration

### Using HuggingFace Datasets

1. Choose a dataset from [HuggingFace Datasets](https://huggingface.co/datasets):
```yaml
dataset:
  type: "huggingface"
  name: "imagenet-1k"  # Replace with your chosen dataset
```

2. Configure dataset parameters:
```yaml
dataset:
  type: "huggingface"
  name: "cifar10"
  batch_size: 128
  num_workers: 4
  image_size: 224
  image_key: "img"      # Check dataset documentation for correct keys
  label_key: "label"
  mean: [0.4914, 0.4822, 0.4465]  # Dataset-specific values
  std: [0.2023, 0.1994, 0.2010]
```

### Using Local Datasets

1. Create a custom DataModule (see dermmnist_datamodule.py as example)
2. Update configuration:
```yaml
dataset:
  type: "your_dataset_type"
  data_dir: "path/to/data"
  batch_size: 32
  num_workers: 4
  image_size: 224
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

## Usage

1. Configure your dataset and model in `configs/config.yaml`
2. Run the training script:
```bash
python benchmark.py
```

The script will:
1. Load the specified dataset (HuggingFace or local)
2. Initialize the chosen model
3. Train using the specified configuration
4. Log metrics and save checkpoints

## Dataset Tips

### HuggingFace Datasets
- Check dataset documentation for correct image/label keys
- Use dataset-specific normalization values when available
- Consider dataset size when setting batch size
- Use streaming for large datasets

### Local Datasets
- Follow the DermMNIST example for custom implementations
- Implement proper data loading and preprocessing
- Handle dataset-specific requirements

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
- Datasets (HuggingFace)
- torchvision
- Weights & Biases
- scikit-learn

For a complete list of dependencies, see `requirements.txt`.

## License

This project is licensed under the MIT License.
