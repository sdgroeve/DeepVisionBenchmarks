from typing import Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchmetrics import Accuracy, Precision, Recall, F1Score

class ImageClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Metrics
        metrics_kwargs = {
            "task": "binary" if num_classes == 2 else "multiclass",
            "num_classes": num_classes
        }
        
        self.train_acc = Accuracy(**metrics_kwargs)
        self.val_acc = Accuracy(**metrics_kwargs)
        self.test_acc = Accuracy(**metrics_kwargs)
        self.test_precision = Precision(**metrics_kwargs)
        self.test_recall = Recall(**metrics_kwargs)
        self.test_f1 = F1Score(**metrics_kwargs)

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)  # Get predicted class indices
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', self.train_acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)  # Get predicted class indices
        self.val_acc(preds, y)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)  # Get predicted class indices
        self.test_acc(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_f1(preds, y)
        
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', self.test_acc, sync_dist=True)
        self.log('test_precision', self.test_precision, sync_dist=True)
        self.log('test_recall', self.test_recall, sync_dist=True)
        self.log('test_f1', self.test_f1, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

def create_model(model_config: Dict[str, Any]) -> ImageClassificationModel:
    """Factory function to create a model instance from config."""
    return ImageClassificationModel(
        model_name=model_config["name"],
        num_classes=model_config.get("num_classes", 2),
        learning_rate=model_config.get("learning_rate", 1e-4),
        weight_decay=model_config.get("weight_decay", 1e-2)
    )
