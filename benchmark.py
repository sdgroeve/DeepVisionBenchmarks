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
            batch_size=config["dataset"]["batch_size"],
            num_workers=config["dataset"]["num_workers"],
            image_size=config["dataset"]["image_size"],
            image_key=config["dataset"].get("image_key", "image"),
            label_key=config["dataset"].get("label_key", "label"),
            mean=config["dataset"].get("mean", (0.485, 0.456, 0.406)),
            std=config["dataset"].get("std", (0.229, 0.224, 0.225))
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

def benchmark_model(model_config: dict, data_module: pl.LightningDataModule, training_config: dict, logging_config: dict):
    """Benchmark a single model configuration."""
    # Create model
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
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_config["max_epochs"],
        accelerator=training_config["accelerator"],
        devices=training_config["devices"],
        strategy=training_config["strategy"],
        precision=training_config["precision"],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=logging_config["log_every_n_steps"]
    )
    
    # Train and test
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Benchmark vision models on various datasets")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    # Load configuration
    config = load_config(args.config)
    
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
        benchmark_model(
            model_config=model_config,
            data_module=data_module,
            training_config=config["training"],
            logging_config=config["logging"]
        )

if __name__ == "__main__":
    main()
