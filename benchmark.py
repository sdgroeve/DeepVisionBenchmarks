import os
import yaml
import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.data.pcam_dataset import PCAMDataModule
from src.models.model_factory import create_model

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def benchmark_model(model_config: dict, data_module: PCAMDataModule, training_config: dict, logging_config: dict):
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
    parser = argparse.ArgumentParser(description="Benchmark image classification models on PCAM dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create data module
    data_module = PCAMDataModule(**config["dataset"])
    
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
