"""
Model training utilities.
"""

import logging
from pathlib import Path
from typing import Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..config import get_config

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train resume-job compatibility models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = None,
        batch_size: int = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        config = get_config()
        
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate or config.model.learning_rate
        self.batch_size = batch_size or config.model.batch_size
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        logger.info(
            f"Initialized ModelTrainer: device={device}, "
            f"lr={self.learning_rate}, batch_size={self.batch_size}"
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'attention' in str(self.model.__class__):
                outputs, _ = self.model(batch)
            else:
                outputs = self.model(batch)
            
            # Normalize targets to 0-1 range
            targets = batch.y / 100.0
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'attention' in str(self.model.__class__):
                    outputs, _ = self.model(batch)
                else:
                    outputs = self.model(batch)
                
                targets = batch.y / 100.0
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(
        self,
        dataset: List,
        epochs: int = None,
        test_size: float = 0.2,
        save_path: Optional[str] = None,
        save_best: bool = True,
        early_stopping_patience: int = 10
    ) -> dict:
        """
        Train model on dataset.
        
        Args:
            dataset: List of HeteroData graphs
            epochs: Number of training epochs
            test_size: Fraction of data for validation
            save_path: Path to save model checkpoints
            save_best: Whether to save best model
            early_stopping_patience: Epochs to wait before early stopping
            
        Returns:
            Training history dictionary
        """
        config = get_config()
        epochs = epochs or config.model.epochs
        
        # Split dataset
        train_dataset, val_dataset = train_test_split(
            dataset,
            test_size=test_size,
            random_state=config.model.random_state
        )
        
        logger.info(
            f"Training on {len(train_dataset)} samples, "
            f"validating on {len(val_dataset)} samples"
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc="Training"):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
                )
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Saved best model with val_loss={val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
        
        return history
    
    def save_model(self, path: str):
        """
        Save model to file.
        
        Args:
            path: Path to save model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model, path)
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path: str, device: str = "cpu") -> nn.Module:
        """
        Load model from file.
        
        Args:
            path: Path to model file
            device: Device to load model on
            
        Returns:
            Loaded model
        """
        model = torch.load(path, map_location=torch.device(device))
        logger.info(f"Model loaded from {path}")
        return model

