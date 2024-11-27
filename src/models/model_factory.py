"""
Model factory module for creating Vision Transformer based image classification models.
This module provides a PyTorch Lightning implementation of image classification models
using pre-trained transformers from HuggingFace, with support for various metrics
and optimized training configurations.

Model Selection Guide:
---------------------
To use a different model from HuggingFace:

1. Model Selection:
   - Visit https://huggingface.co/models?pipeline_tag=image-classification
   - Choose any model that supports image classification
   - Common architectures include:
     * Vision Transformers (ViT)
     * Swin Transformers
     * DeiT
     * ConvNeXT
     * BEiT

2. Configuration Updates:
   - In configs/config.yaml, update the model section:
     ```yaml
     models:
       - name: "your-chosen-model"  # HuggingFace model identifier
         num_classes: 7             # Keep as 7 for DermMNIST
         learning_rate: 0.0001      # Adjust if needed
         weight_decay: 0.01         # Adjust if needed
     ```

3. Image Size Requirements:
   - Check the model's documentation for required input size
   - Update dataset.image_size in config.yaml to match
   - Common sizes: 224x224, 384x384, 512x512

4. Training Optimization:
   - Adjust learning_rate based on model size:
     * Smaller models (tiny/small): 1e-4 to 5e-4
     * Larger models (base/large): 1e-5 to 1e-4
   - Adjust weight_decay based on training behavior:
     * Increase if overfitting (e.g., 0.1)
     * Decrease if underfitting (e.g., 0.001)

Example Configurations:
---------------------
1. ViT Base (current default):
   ```yaml
   - name: "google/vit-base-patch16-224"
     num_classes: 7
     learning_rate: 0.0001
     weight_decay: 0.01
   ```

2. Swin Transformer:
   ```yaml
   - name: "microsoft/swin-tiny-patch4-window7-224"
     num_classes: 7
     learning_rate: 0.0001
     weight_decay: 0.01
   ```

3. DeiT:
   ```yaml
   - name: "facebook/deit-base-patch16-224"
     num_classes: 7
     learning_rate: 0.0001
     weight_decay: 0.01
   ```

4. ConvNeXT:
   ```yaml
   - name: "facebook/convnext-tiny-224"
     num_classes: 7
     learning_rate: 0.0001
     weight_decay: 0.01
   ```

Note: The ImageClassificationModel class automatically handles:
- Model architecture differences
- Input preprocessing
- Output layer adaptation
- Optimizer configuration
"""

from typing import Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchmetrics import Accuracy, Precision, Recall, F1Score

class ImageClassificationModel(pl.LightningModule):
    """
    PyTorch Lightning module for image classification using pre-trained transformer models.
    
    This class implements a complete training pipeline including:
    - Model initialization from HuggingFace pre-trained weights
    - Training, validation, and test steps
    - Metrics computation and logging
    - Optimizer and learning rate scheduler configuration
    
    The class is designed to work with any image classification model from HuggingFace's
    model hub. To use a different model, simply update the model_name parameter with
    the desired model identifier from HuggingFace.
    
    Attributes:
        processor: HuggingFace image processor for input preprocessing
        model: Pre-trained transformer model for image classification
        train_acc: Training accuracy metric
        val_acc: Validation accuracy metric
        test_acc: Test accuracy metric
        test_precision: Test precision metric
        test_recall: Test recall metric
        test_f1: Test F1 score metric
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        """
        Initialize the image classification model.
        
        Args:
            model_name (str): HuggingFace model identifier (e.g., "google/vit-base-patch16-224")
            num_classes (int): Number of target classes (default: 2)
            learning_rate (float): Initial learning rate (default: 1e-4)
            weight_decay (float): Weight decay for AdamW optimizer (default: 1e-2)
            
        Note:
            When changing models, ensure that:
            1. The model_name is valid and exists on HuggingFace
            2. The input image size matches the model's requirements
            3. The learning_rate and weight_decay are appropriate for the model
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # Handles input size mismatches
        )
        
        # Initialize metrics for model evaluation
        metrics_kwargs = {
            "task": "binary" if num_classes == 2 else "multiclass",
            "num_classes": num_classes
        }
        
        # Create metrics for different stages
        self.train_acc = Accuracy(**metrics_kwargs)
        self.val_acc = Accuracy(**metrics_kwargs)
        self.test_acc = Accuracy(**metrics_kwargs)
        self.test_precision = Precision(**metrics_kwargs)
        self.test_recall = Recall(**metrics_kwargs)
        self.test_f1 = F1Score(**metrics_kwargs)

    def forward(self, x):
        """Forward pass of the model."""
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        """Training step for a single batch."""
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update and log metrics
        self.train_acc(preds, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', self.train_acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for a single batch."""
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update and log metrics
        self.val_acc(preds, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for a single batch."""
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update all test metrics
        self.test_acc(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_f1(preds, y)
        
        # Log all test metrics
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', self.test_acc, sync_dist=True)
        self.log('test_precision', self.test_precision, sync_dist=True)
        self.log('test_recall', self.test_recall, sync_dist=True)
        self.log('test_f1', self.test_f1, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Note:
            When using a different model, you may need to adjust:
            1. Learning rate based on model size
            2. Weight decay based on overfitting/underfitting
            3. Scheduler parameters based on training behavior
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing scheduler for learning rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,  # Maximum number of iterations
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"  # Monitor validation loss for scheduling
            }
        }

def create_model(model_config: Dict[str, Any]) -> ImageClassificationModel:
    """
    Factory function to create a model instance from configuration.
    
    Args:
        model_config (Dict[str, Any]): Configuration dictionary containing:
            - name: HuggingFace model identifier
            - num_classes: Number of target classes
            - learning_rate: Initial learning rate
            - weight_decay: Weight decay for optimizer
    
    Returns:
        ImageClassificationModel: Initialized model instance
        
    Example:
        ```python
        config = {
            "name": "google/vit-base-patch16-224",
            "num_classes": 7,
            "learning_rate": 0.0001,
            "weight_decay": 0.01
        }
        model = create_model(config)
        ```
    """
    return ImageClassificationModel(
        model_name=model_config["name"],
        num_classes=model_config.get("num_classes", 2),
        learning_rate=model_config.get("learning_rate", 1e-4),
        weight_decay=model_config.get("weight_decay", 1e-2)
    )
