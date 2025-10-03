"""
Graph Neural Network models for resume-job compatibility prediction.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, GATConv, HeteroConv

from ..config import get_config

logger = logging.getLogger(__name__)


class ResumeJobGNN(nn.Module):
    """
    Heterogeneous GNN for resume-job compatibility.
    Uses GraphConv for job-skill edges and GAT for resume-skill edges.
    """
    
    def __init__(
        self,
        embedding_dim: int = 1536,
        hidden_dims: list = None,
        gat_heads: int = 8,
        dropout: float = 0.2
    ):
        """
        Initialize model.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions for MLP
            gat_heads: Number of attention heads for GAT
            dropout: Dropout rate
        """
        super(ResumeJobGNN, self).__init__()
        
        config = get_config()
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or config.model.hidden_dims
        self.gat_heads = gat_heads
        self.dropout_rate = dropout
        
        # Heterogeneous graph convolution layer
        self.conv_job = GraphConv(embedding_dim, embedding_dim)
        self.conv_resume = GATConv(
            embedding_dim,
            embedding_dim,
            heads=gat_heads,
            concat=False,  # Average attention heads
            add_self_loops=False
        )
        
        # MLP for regression
        mlp_layers = []
        input_dim = embedding_dim * 2  # Concatenated job + resume embeddings
        
        for hidden_dim in self.hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer
        mlp_layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        logger.info(
            f"Initialized ResumeJobGNN: embedding_dim={embedding_dim}, "
            f"hidden_dims={self.hidden_dims}, gat_heads={gat_heads}"
        )
    
    def forward(self, data: HeteroData):
        """
        Forward pass.
        
        Args:
            data: HeteroData graph
            
        Returns:
            Tuple of (score, attention_scores)
        """
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        # Apply graph convolutions
        x_job = self.conv_job(
            (x_dict['skill'], x_dict['job']),
            edge_index_dict[('skill', 'to', 'job')]
        )
        
        x_resume, attn_weights = self.conv_resume(
            (x_dict['skill'], x_dict['resume']),
            edge_index_dict[('skill', 'to', 'resume')],
            return_attention_weights=True
        )
        
        # Apply activation
        x_job = F.leaky_relu(x_job)
        x_resume = F.leaky_relu(x_resume)
        
        # Concatenate job and resume embeddings
        combined = torch.cat([x_job, x_resume], dim=1)
        
        # Pass through MLP
        score = self.mlp(combined).squeeze()
        
        # Store attention scores for explainability
        attention_scores = {
            'skill_to_resume': attn_weights
        }
        
        return score, attention_scores


class SimpleGCNModel(nn.Module):
    """
    Simpler GNN model using only GraphConv layers.
    Faster but without attention mechanism.
    """
    
    def __init__(
        self,
        embedding_dim: int = 1536,
        hidden_dims: list = None,
        dropout: float = 0.2
    ):
        """
        Initialize simple GCN model.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions for MLP
            dropout: Dropout rate
        """
        super(SimpleGCNModel, self).__init__()
        
        config = get_config()
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or config.model.hidden_dims
        
        # Heterogeneous convolution
        self.conv = HeteroConv({
            ('skill', 'to', 'job'): GraphConv(embedding_dim, embedding_dim),
            ('skill', 'to', 'resume'): GraphConv(embedding_dim, embedding_dim),
        }, aggr='mean')
        
        # MLP for regression
        mlp_layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in self.hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        
        logger.info(f"Initialized SimpleGCNModel: embedding_dim={embedding_dim}")
    
    def forward(self, data: HeteroData):
        """
        Forward pass.
        
        Args:
            data: HeteroData graph
            
        Returns:
            Compatibility score (no attention scores)
        """
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        # Apply heterogeneous convolution
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Concatenate embeddings
        combined = torch.cat([x_dict['job'], x_dict['resume']], dim=1)
        
        # Predict score
        score = self.mlp(combined).squeeze()
        
        return score

