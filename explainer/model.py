import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import HeteroData



class SimpleGAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h):
        super(SimpleGAT, self).__init__()

        # Separate GAT layers for each edge type
        self.conv_job = GraphConv(1536, 1536)
        self.conv_resume = GATConv(1536, 1536, heads=8, concat=True, add_self_loops=False)

        # Define linear layers for regression task
        self.hidden_layer0 = nn.Linear(13824, 12000)
        self.hidden_layerE = nn.Linear(12000, 6000)
        self.hidden_layerE2 = nn.Linear(6000, 3072)
        self.hidden_layerE3 = nn.Linear(3072, 2048)
        self.hidden_layer1 = nn.Linear(2048, 1024)
        self.hidden_layer2 = nn.Linear(1024, 512)
        self.hidden_layer3 = nn.Linear(512, 256)
        self.hidden_layer4 = nn.Linear(256, 128)
        self.hidden_layer5 = nn.Linear(128, 64)
        self.score_predictor = nn.Linear(64, 1)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        attention_scores = {}

        # Compute GATConv with attention scores for each edge type separately
        x_skill_to_job = self.conv_job(
            (x_dict['skill'], x_dict['job']), edge_index_dict[('skill', 'to', 'job')]
        )
        x_skill_to_resume, attn_resume = self.conv_resume(
            (x_dict['skill'], x_dict['resume']), edge_index_dict[('skill', 'to', 'resume')], return_attention_weights=True
        )

        # Save updated node embeddings and attention scores
        x_dict['job'] = x_skill_to_job
        x_dict['resume'] = x_skill_to_resume
        #attention_scores['skill_to_job'] = attn_job
        attention_scores['skill_to_resume'] = attn_resume

        # Activation
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        # Concatenate 'job' and 'resume' node embeddings
        combined_embedding = torch.cat((x_dict['job'], x_dict['resume']), dim=1)

        # Forward pass through linear layers
        x = F.leaky_relu(self.hidden_layer0(combined_embedding))
        x = F.leaky_relu(self.hidden_layerE(x))
        x = F.leaky_relu(self.hidden_layerE2(x))
        x = F.leaky_relu(self.hidden_layerE3(x))
        x = F.leaky_relu(self.hidden_layer1(x))
        x = F.leaky_relu(self.hidden_layer2(x))
        x = F.leaky_relu(self.hidden_layer3(x))
        x = F.leaky_relu(self.hidden_layer4(x))
        x = F.leaky_relu(self.hidden_layer5(x))
        score = self.score_predictor(x).squeeze()

        return score, attention_scores
