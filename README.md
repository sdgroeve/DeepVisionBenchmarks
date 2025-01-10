# DeepVision: Flexible Vision Model Training Framework

A PyTorch Lightning-based framework for training and evaluating various deep learning models on image classification tasks. The framework supports both HuggingFace datasets and custom local datasets.

## Project Overview

This project implements a flexible deep learning pipeline that can work with:
- Any image classification model from HuggingFace
- Any dataset from HuggingFace's datasets hub
- Local custom datasets through custom DataModules

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
The framework provides a flexible base class for implementing custom local dataset loaders. You can create your own dataset module by inheriting from `LocalImageDataModule`.

#### Using Existing Local Datasets

Example configuration for DermMNIST:
```yaml
dataset:
  type: "dermmnist"
  data_dir: "data/dermmnist"
  batch_size: 32
  num_workers: 4
  image_size: 224
```

#### Creating Custom Dataset Modules

1. Create a custom dataset class inheriting from `LocalImageDataset`:
```python
from src.data.local_datamodule import LocalImageDataset

class CustomDataset(LocalImageDataset):
    def _load_dataset(self):
        # Implement your dataset loading logic
        samples = []
        # Example: Loading from a directory structure
        for class_idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_idx))
        return samples
```

2. Create a DataModule inheriting from `LocalImageDataModule`:
```python
from src.data.local_datamodule import LocalImageDataModule

class CustomDataModule(LocalImageDataModule):
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(
            data_dir=data_dir,
            **kwargs,
            # Specify dataset-specific settings
            mean=[0.485, 0.456, 0.406],  # Dataset mean
            std=[0.229, 0.224, 0.225],   # Dataset std
            num_channels=3               # Number of channels
        )
    
    def _create_dataset(self, split: str):
        return CustomDataset(
            root_dir=self.data_dir,
            transform=self.transform,
            split=split
        )
```

3. Update configuration:
```yaml
dataset:
  type: "custom"  # Register your dataset type
  data_dir: "path/to/data"
  batch_size: 32
  num_workers: 4
  image_size: 224
```

See `src/data/dermmnist_datamodule.py` for a complete example implementation.

#### Dataset Structure Tips

1. Organize your data in a clear directory structure:
```
data/
└── your_dataset/
    ├── train/
    │   ├── class1/
    │   │   ├── image1.jpg
    │   │   └── image2.jpg
    │   └── class2/
    ├── val/
    └── test/
```

2. Consider dataset-specific requirements:
   - Image formats and channels
   - Normalization values
   - Data augmentation needs
   - Special handling for labels

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
- Some datasets might require authentication - use `huggingface-cli login`

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
- Weights & Bibes
- scikit-learn

For a complete list of dependencies, see `requirements.txt`.

## Troubleshooting

1. ModuleNotFoundError: No module named 'datasets'
   - Solution: Reinstall requirements with `pip install --upgrade -r requirements.txt`

2. Dataset access issues:
   - Some datasets might require authentication
   - Run `huggingface-cli login` and follow the prompts
   - Visit [HuggingFace](https://huggingface.co/) to create an account if needed

## License

This project is licensed under the MIT License.
