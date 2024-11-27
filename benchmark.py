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
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_gpu():
    """Check GPU availability and display information."""
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

def benchmark_model(model_config: dict, data_module: DermMNISTDataModule, training_config: dict, logging_config: dict):
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
    parser = argparse.ArgumentParser(description="Benchmark image classification models on DermMNIST dataset")
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
    data_module = DermMNISTDataModule(**config["dataset"])
    
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
