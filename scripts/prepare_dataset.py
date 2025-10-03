#!/usr/bin/env python3
"""
Script to prepare dataset from MongoDB and save as .pt file.
"""

import argparse
import logging
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, set_config
from src.graph.builder import GraphBuilder
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/dataset.pt",
        help="Output path for dataset (.pt file)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)"
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
        name="prepare_dataset",
        level=getattr(logging, args.log_level),
        log_file="logs/prepare_dataset.log"
    )
    
    try:
        # Initialize configuration
        logger.info("Loading configuration...")
        config = Config.from_env()
        set_config(config)
        
        # Initialize graph builder
        logger.info("Initializing graph builder...")
        builder = GraphBuilder()
        
        # Build dataset
        logger.info("Building dataset from MongoDB...")
        dataset = builder.build_dataset(limit=args.limit, show_progress=True)
        
        # Save dataset
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(dataset, args.output)
        logger.info(f"Dataset saved to {args.output} ({len(dataset)} samples)")
        
        # Print statistics
        scores = [data.y.item() for data in dataset]
        logger.info(f"Score statistics:")
        logger.info(f"  Min: {min(scores):.2f}")
        logger.info(f"  Max: {max(scores):.2f}")
        logger.info(f"  Mean: {sum(scores)/len(scores):.2f}")
        
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

