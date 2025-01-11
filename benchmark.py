"""
Benchmark script for training and evaluating vision models on various datasets.
Supports both local datasets and datasets from HuggingFace's hub.
"""

import os
import yaml
import argparse
from pathlib import Path
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.models.model_factory import create_model
from src.data.dermmnist_datamodule import DermMNISTDataModule
from src.data.huggingface_datamodule import HuggingFaceDataModule

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_gpu():
    """
    Check GPU availability and display information.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    print("\nChecking GPU availability...")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("WARNING: No GPU detected! Training will be significantly slower on CPU.")
        print("Consider using a machine with a CUDA-capable GPU for faster training.")
        return False

def create_data_module(config: dict):
    """
    Create the appropriate data module based on configuration.
    
    Args:
        config (dict): Dataset configuration dictionary
        
    Returns:
        pl.LightningDataModule: Configured data module
    """
    dataset_type = config["dataset"].get("type", "dermmnist")
    
    if dataset_type == "huggingface":
        # Create HuggingFace dataset module
        return HuggingFaceDataModule(
            dataset_name=config["dataset"]["name"],
            model_name=config["models"][0]["name"],  # Use first model's name
            batch_size=config["dataset"]["batch_size"],
            num_workers=config["dataset"]["num_workers"],
            image_key=config["dataset"].get("image_key", "image"),
            label_key=config["dataset"].get("label_key", "label"),
            mean=config["dataset"].get("mean", (0.485, 0.456, 0.406)),
            std=config["dataset"].get("std", (0.229, 0.224, 0.225)),
            num_channels=config["dataset"].get("num_channels", 3)
        )
    elif dataset_type == "dermmnist":
        # Create DermMNIST dataset module
        return DermMNISTDataModule(
            data_dir=config["dataset"]["data_dir"],
            batch_size=config["dataset"]["batch_size"],
            num_workers=config["dataset"]["num_workers"],
            image_size=config["dataset"]["image_size"]
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def add_model_args(parser):
    """Add model-specific command line arguments."""
    model_args = parser.add_argument_group('Model Configuration')
    model_args.add_argument("--head-dropout", type=float, 
                           help="Dropout rate for classification head (default: from config)")
    model_args.add_argument("--head-hidden-sizes", type=int, nargs="+", 
                           help="Hidden layer sizes for custom head (default: from config)")
    model_args.add_argument("--label-smoothing", type=float, 
                           help="Label smoothing factor 0.0-1.0 (default: from config)")
    model_args.add_argument("--mixup-alpha", type=float, 
                           help="Mixup alpha parameter, 0.0 to disable (default: from config)")
    model_args.add_argument("--progressive-unfreezing", action="store_true", 
                           help="Enable progressive layer unfreezing")
    model_args.add_argument("--discriminative-lr", action="store_true", 
                           help="Use different learning rates for layers")
    model_args.add_argument("--head-lr-multiplier", type=float, 
                           help="Learning rate multiplier for head (default: from config)")
    model_args.add_argument("--gradient-clip-val", type=float, 
                           help="Gradient clipping value, 0.0 to disable (default: from config)")
    return parser

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    # Only update if argument is provided
    for model in config["models"]:
        if args.head_dropout is not None:
            model["head_dropout"] = args.head_dropout
        if args.head_hidden_sizes is not None:
            model["head_hidden_sizes"] = args.head_hidden_sizes
        if args.label_smoothing is not None:
            model["label_smoothing"] = args.label_smoothing
        if args.mixup_alpha is not None:
            model["mixup_alpha"] = args.mixup_alpha
        if args.progressive_unfreezing:
            model["progressive_unfreezing"] = True
        if args.discriminative_lr:
            model["discriminative_lr"] = True
        if args.head_lr_multiplier is not None:
            model["head_lr_multiplier"] = args.head_lr_multiplier
        if args.gradient_clip_val is not None:
            model["gradient_clip_val"] = args.gradient_clip_val
    return config

def benchmark_model(model_config: dict, data_module: pl.LightningDataModule, training_config: dict, logging_config: dict):
    """Benchmark a single model configuration."""
    # Create model with advanced training features
    model = create_model(model_config)
    
    # Set up logging
    logger = WandbLogger(
        project=logging_config["project_name"],
        name=f"benchmark-{model_config['name'].split('/')[-1]}",
        save_dir=logging_config["save_dir"]
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(logging_config["save_dir"], "checkpoints"),
            filename=f"{model_config['name'].split('/')[-1]}" + "-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        )
    ]
    
    # Create trainer with gradient clipping if enabled
    trainer_kwargs = {
        "max_epochs": training_config["max_epochs"],
        "accelerator": training_config["accelerator"],
        "devices": training_config["devices"],
        "strategy": training_config["strategy"],
        "precision": training_config["precision"],
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": logging_config["log_every_n_steps"]
    }
    
    # Add gradient clipping if specified
    if model_config.get("gradient_clip_val", 0.0) > 0:
        trainer_kwargs["gradient_clip_val"] = model_config["gradient_clip_val"]
        trainer_kwargs["gradient_clip_algorithm"] = "norm"
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train and test
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Benchmark vision models on various datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="Path to configuration file")
    parser = add_model_args(parser)
    args = parser.parse_args()
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Adjust configuration if no GPU is available
    if not has_gpu and config["training"]["accelerator"] == "gpu":
        print("\nAdjusting configuration to use CPU instead of GPU...")
        config["training"]["accelerator"] = "cpu"
        config["training"]["devices"] = None
    
    # Create data module
    data_module = create_data_module(config)
    
    # Benchmark each model
    for model_config in config["models"]:
        print(f"\nBenchmarking model: {model_config['name']}")
        print("\nAdvanced training features:")
        if model_config.get("head_hidden_sizes"):
            print(f"- Custom head with layers: {model_config['head_hidden_sizes']}")
        if model_config.get("progressive_unfreezing"):
            print("- Progressive layer unfreezing enabled")
        if model_config.get("discriminative_lr"):
            print(f"- Discriminative learning rates (head multiplier: {model_config.get('head_lr_multiplier', 10.0)})")
        if model_config.get("mixup_alpha", 0) > 0:
            print(f"- Mixup augmentation (alpha: {model_config['mixup_alpha']})")
        if model_config.get("label_smoothing", 0) > 0:
            print(f"- Label smoothing: {model_config['label_smoothing']}")
        if model_config.get("gradient_clip_val", 0) > 0:
            print(f"- Gradient clipping: {model_config['gradient_clip_val']}")
        
        benchmark_model(
            model_config=model_config,
            data_module=data_module,
            training_config=config["training"],
            logging_config=config["logging"]
        )

if __name__ == "__main__":
    main()
