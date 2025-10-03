"""
Model evaluation utilities.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate resume-job compatibility models."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        logger.info(f"Initialized ModelEvaluator on device: {device}")
    
    def predict(self, data: HeteroData) -> Tuple[float, dict]:
        """
        Make prediction for a single graph.
        
        Args:
            data: HeteroData graph
            
        Returns:
            Tuple of (predicted_score, attention_scores)
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                # Try to get attention scores if available
                try:
                    output, attention_scores = self.model(data)
                    predicted_score = output.item() * 100  # Denormalize
                    return predicted_score, attention_scores
                except (ValueError, TypeError):
                    # Model doesn't return attention scores
                    output = self.model(data)
                    predicted_score = output.item() * 100
                    return predicted_score, {}
    
    def evaluate(self, test_dataset: List[HeteroData]) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_dataset: List of test graphs
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = []
        ground_truth = []
        
        logger.info(f"Evaluating on {len(test_dataset)} samples...")
        
        for data in test_dataset:
            pred_score, _ = self.predict(data)
            true_score = data.y.item()
            
            predictions.append(pred_score)
            ground_truth.append(true_score)
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        mae = mean_absolute_error(ground_truth, predictions)
        mse = mean_squared_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(ground_truth, predictions)
        
        # Normalize metrics to 0-1 scale
        mae_normalized = mae / 100.0
        mse_normalized = mse / (100.0 ** 2)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae_normalized': mae_normalized,
            'mse_normalized': mse_normalized
        }
        
        logger.info(
            f"Evaluation Results: "
            f"MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}"
        )
        
        return metrics
    
    def evaluate_with_tolerance(
        self,
        test_dataset: List[HeteroData],
        tolerance: float = 5.0
    ) -> float:
        """
        Evaluate accuracy with tolerance (as in thesis).
        
        Args:
            test_dataset: List of test graphs
            tolerance: Acceptable error margin (default: ±5 points)
            
        Returns:
            Accuracy (percentage of predictions within tolerance)
        """
        correct = 0
        total = len(test_dataset)
        
        for data in test_dataset:
            pred_score, _ = self.predict(data)
            true_score = data.y.item()
            
            if abs(pred_score - true_score) <= tolerance:
                correct += 1
        
        accuracy = (correct / total) * 100
        
        logger.info(
            f"Accuracy (±{tolerance} tolerance): {accuracy:.2f}% "
            f"({correct}/{total} correct)"
        )
        
        return accuracy
    
    def get_attention_scores(self, data: HeteroData) -> Dict[str, np.ndarray]:
        """
        Get attention scores for a graph (for explainability).
        
        Args:
            data: HeteroData graph
            
        Returns:
            Dictionary of attention scores
        """
        _, attention_scores = self.predict(data)
        
        if not attention_scores:
            logger.warning("Model does not provide attention scores")
            return {}
        
        # Extract and format attention scores
        formatted_scores = {}
        
        if 'skill_to_resume' in attention_scores:
            edge_index, edge_weights = attention_scores['skill_to_resume']
            
            # Get max attention for each skill
            max_attention = torch.max(edge_weights, dim=1).values.cpu().numpy()
            skill_titles = data['skill'].title if hasattr(data['skill'], 'title') else []
            
            formatted_scores['skills'] = [
                {'skill': title, 'attention': float(attn)}
                for title, attn in zip(skill_titles, max_attention)
            ]
        
        return formatted_scores

