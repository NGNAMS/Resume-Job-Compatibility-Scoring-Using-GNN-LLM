#!/usr/bin/env python3
"""
Script to train resume-job compatibility model.
"""

import argparse
import logging
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, set_config
from src.models.gnn_models import ResumeJobGNN, SimpleGCNModel
from src.models.trainer import ModelTrainer
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Train resume-job compatibility model")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (.pt file)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gnn",
        choices=["gnn", "gcn"],
        help="Model architecture (gnn=GAT+GraphConv, gcn=GraphConv only)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=3072,
        choices=[1536, 3072],
        help="Embedding dimension (1536 for small, 3072 for large)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model.pth",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(
        name="train_model",
        level=getattr(logging, args.log_level),
        log_file="logs/train_model.log"
    )
    
    try:
        # Initialize configuration
        logger.info("Loading configuration...")
        config = Config.from_env()
        set_config(config)
        
        # Load dataset
        logger.info(f"Loading dataset from {args.dataset}...")
        dataset = torch.load(args.dataset)
        logger.info(f"Loaded {len(dataset)} samples")
        
        # Initialize model
        logger.info(f"Initializing {args.model_type} model...")
        if args.model_type == "gnn":
            model = ResumeJobGNN(embedding_dim=args.embedding_dim)
        else:
            model = SimpleGCNModel(embedding_dim=args.embedding_dim)
        
        # Initialize trainer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training on device: {device}")
        
        trainer = ModelTrainer(
            model=model,
            device=device,
            learning_rate=args.lr,
            batch_size=args.batch_size
        )
        
        # Train
        logger.info("Starting training...")
        history = trainer.train(
            dataset=dataset,
            epochs=args.epochs,
            save_path=args.output,
            save_best=True,
            early_stopping_patience=10
        )
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")
        logger.info(f"Model saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

