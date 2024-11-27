"""
Benchmark script for training Vision Transformer models on the DermMNIST dataset.
This script handles model training, evaluation, and logging using PyTorch Lightning.
It includes GPU detection and automatic hardware configuration adjustment.
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

def load_config(config_path: str) -> dict:
    """
    Load and parse the YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Parsed configuration dictionary containing model, training, and logging settings
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_gpu() -> bool:
    """
    Check GPU availability and display detailed hardware information.
    
    This function:
    1. Verifies if CUDA is available through PyTorch
    2. If GPU is available:
       - Counts available GPUs
       - Lists each GPU's name/model
       - Displays CUDA version
    3. If no GPU is found, provides a warning about performance implications
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    print("\nChecking GPU availability...")
    if torch.cuda.is_available():
        # Get count of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s):")
        
        # Display information for each available GPU
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA Version: {torch.version.cuda}")
        return True
    else:
        # Warning messages for CPU-only setup
        print("WARNING: No GPU detected! Training will be significantly slower on CPU.")
        print("Consider using a machine with a CUDA-capable GPU for faster training.")
        return False

def benchmark_model(model_config: dict, data_module: DermMNISTDataModule, training_config: dict, logging_config: dict):
    """
    Train and evaluate a model with the specified configuration.
    
    This function handles:
    1. Model initialization
    2. Setting up logging with Weights & Biases
    3. Configuring checkpointing and early stopping
    4. Training and testing the model
    
    Args:
        model_config (dict): Model-specific configuration (architecture, hyperparameters)
        data_module (DermMNISTDataModule): PyTorch Lightning data module for DermMNIST
        training_config (dict): Training settings (epochs, hardware, precision)
        logging_config (dict): Logging configuration (project name, save directory)
    """
    # Initialize the model using the factory pattern
    model = create_model(model_config)
    
    # Configure Weights & Biases logging
    logger = WandbLogger(
        project=logging_config["project_name"],
        name=f"benchmark-{model_config['name'].split('/')[-1]}",
        save_dir=logging_config["save_dir"]
    )
    
    # Set up training callbacks
    callbacks = [
        # Model checkpointing configuration
        ModelCheckpoint(
            dirpath=os.path.join(logging_config["save_dir"], "checkpoints"),
            filename=f"{model_config['name'].split('/')[-1]}" + "-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1  # Save only the best model
        ),
        # Early stopping configuration
        EarlyStopping(
            monitor="val_loss",
            patience=3,  # Stop if no improvement for 3 epochs
            mode="min"
        )
    ]
    
    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=training_config["max_epochs"],
        accelerator=training_config["accelerator"],  # GPU/CPU selection
        devices=training_config["devices"],          # Number of devices to use
        strategy=training_config["strategy"],        # Training strategy (DP/DDP)
        precision=training_config["precision"],      # Numerical precision
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=logging_config["log_every_n_steps"]
    )
    
    # Execute training and testing
    trainer.fit(model, data_module)    # Train the model
    trainer.test(model, data_module)   # Evaluate on test set

def main():
    """
    Main execution function that:
    1. Parses command line arguments
    2. Checks GPU availability
    3. Loads and adjusts configuration based on hardware
    4. Initializes data module
    5. Runs benchmarking for each model configuration
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark image classification models on DermMNIST dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Check hardware capabilities
    has_gpu = check_gpu()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Adjust configuration if GPU is not available
    if not has_gpu and config["training"]["accelerator"] == "gpu":
        print("\nAdjusting configuration to use CPU instead of GPU...")
        config["training"]["accelerator"] = "cpu"
        config["training"]["devices"] = None  # Remove device specification for CPU
    
    # Initialize data module with dataset configuration
    data_module = DermMNISTDataModule(**config["dataset"])
    
    # Run benchmarking for each model in configuration
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
