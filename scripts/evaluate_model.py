#!/usr/bin/env python3
"""
Script to evaluate trained model.
"""

import argparse
import logging
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, set_config
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate resume-job compatibility model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pth file)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (.pt file)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Tolerance for accuracy metric (±N points)"
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
        name="evaluate_model",
        level=getattr(logging, args.log_level),
        log_file="logs/evaluate_model.log"
    )
    
    try:
        # Initialize configuration
        logger.info("Loading configuration...")
        config = Config.from_env()
        set_config(config)
        
        # Load dataset
        logger.info(f"Loading dataset from {args.dataset}...")
        dataset = torch.load(args.dataset)
        
        # Split dataset
        _, test_dataset = train_test_split(
            dataset,
            test_size=args.test_size,
            random_state=42
        )
        logger.info(f"Testing on {len(test_dataset)} samples")
        
        # Load model
        logger.info(f"Loading model from {args.model}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ModelTrainer.load_model(args.model, device=device)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model=model, device=device)
        
        # Evaluate
        logger.info("Evaluating model...")
        metrics = evaluator.evaluate(test_dataset)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Mean Absolute Error (MAE):  {metrics['mae']:.4f}")
        print(f"Mean Squared Error (MSE):   {metrics['mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"R² Score:                   {metrics['r2']:.4f}")
        print(f"\nNormalized (0-1 scale):")
        print(f"  MAE: {metrics['mae_normalized']:.4f}")
        print(f"  MSE: {metrics['mse_normalized']:.4f}")
        
        # Accuracy with tolerance
        accuracy = evaluator.evaluate_with_tolerance(test_dataset, tolerance=args.tolerance)
        print(f"\nAccuracy (±{args.tolerance} tolerance): {accuracy:.2f}%")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

