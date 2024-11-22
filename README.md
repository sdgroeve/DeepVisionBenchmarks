# DeepVision: Image Classification Benchmark

A modular PyTorch Lightning-based framework for benchmarking state-of-the-art image classification models from HuggingFace on the PCAM dataset.

## Project Structure

```
deepvision/
├── src/
│   ├── data/          # Dataset handling
│   ├── models/        # Model implementations
│   ├── training/      # Training pipeline
│   └── utils/         # Utility functions
├── configs/           # Configuration files
├── benchmark.py       # Main benchmarking script
├── requirements.txt   # Project dependencies
└── README.md         # Documentation
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

## Dataset

The PatchCamelyon (PCAM) dataset is used for benchmarking. It consists of 327,680 color images (96x96px) extracted from histopathologic scans of lymph node sections. This binary classification task involves detecting metastatic tissue in these images.

The dataset will be automatically downloaded and prepared for training when running the benchmark script.

## Available Models

The framework supports various state-of-the-art models from HuggingFace's model hub, including:

- ViT (Vision Transformer)
- DeiT
- Swin Transformer
- ConvNeXt
- EfficientNet
- ResNet

## Usage

Run the benchmark script with default configuration:
```bash
python benchmark.py
```

Specify a custom configuration file:
```bash
python benchmark.py --config configs/custom_config.yaml
```

## Configuration

The `configs/config.yaml` file contains all configurable parameters:

- Dataset parameters
- Model selection
- Training hyperparameters
- Logging settings
- Evaluation metrics

## Results

Benchmark results will be saved in the `results/` directory, including:

- Model performance metrics
- Training curves
- Confusion matrices
- Per-class accuracy statistics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
