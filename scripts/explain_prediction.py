#!/usr/bin/env python3
"""
Script to explain model predictions using attention scores.
"""

import argparse
import logging
from pathlib import Path
import torch
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, set_config
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.graph.builder import GraphBuilder
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Explain model predictions")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pth file)"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="MongoDB ObjectId of job"
    )
    parser.add_argument(
        "--resume-id",
        type=str,
        required=True,
        help="MongoDB ObjectId of resume"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/attention_scores.png",
        help="Output path for visualization"
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
        name="explain_prediction",
        level=getattr(logging, args.log_level),
        log_file="logs/explain_prediction.log"
    )
    
    try:
        # Initialize configuration
        logger.info("Loading configuration...")
        config = Config.from_env()
        set_config(config)
        
        # Build graph
        logger.info("Building graph...")
        builder = GraphBuilder()
        data = builder.build_graph(job_id=args.job_id, resume_id=args.resume_id)
        
        # Load model
        logger.info(f"Loading model from {args.model}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ModelTrainer.load_model(args.model, device=device)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model=model, device=device)
        
        # Get prediction and attention scores
        logger.info("Generating prediction and attention scores...")
        predicted_score, attention_scores = evaluator.predict(data)
        
        # Get formatted attention scores
        formatted_scores = evaluator.get_attention_scores(data)
        
        # Print results
        print("\n" + "="*50)
        print("PREDICTION EXPLANATION")
        print("="*50)
        print(f"Predicted Compatibility Score: {predicted_score:.2f}/100")
        
        if 'skills' in formatted_scores:
            print("\nSkill Importance (Attention Scores):")
            print("-" * 50)
            
            # Sort by attention score
            skills = sorted(
                formatted_scores['skills'],
                key=lambda x: x['attention'],
                reverse=True
            )
            
            for item in skills:
                print(f"  {item['skill']:<30} {item['attention']:.4f}")
            
            # Create visualization
            logger.info("Creating visualization...")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            skill_names = [s['skill'] for s in skills]
            attention_values = [s['attention'] for s in skills]
            
            ax.barh(skill_names, attention_values, color='skyblue')
            ax.set_xlabel('Attention Score')
            ax.set_title('Skill Importance for Resume-Job Match')
            ax.invert_yaxis()
            
            # Save plot
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {args.output}")
            
        else:
            print("\nNote: Model does not provide attention scores for explainability.")
        
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error explaining prediction: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

