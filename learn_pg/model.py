import torch
import pickle
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data, HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric import utils
from torch_geometric.nn import GCNConv, GATConv, GraphConv, HeteroConv, SAGEConv, RGCNConv
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from itertools import count
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer, AttentionExplainer,

class SimpleGCN(torch.nn.Module):

    def __init__(self, dim_in, dim_h):
        super(SimpleGCN, self).__init__()
        self.conv = HeteroConv({
            ('skill', 'to', 'job'): GraphConv(1536, 1536),
            ('skill', 'to', 'resume'): GraphConv(1536, 1536),
        }, aggr='mean')

        self.hidden_layer1 = torch.nn.Linear(3072, 2048)
        self.hidden_layer2 = torch.nn.Linear(2048, 1024)
        self.hidden_layer3 = torch.nn.Linear(1024, 512)
        self.hidden_layer4 = torch.nn.Linear(512, 256)
        self.hidden_layer5 = torch.nn.Linear(256, 128)
        self.hidden_layer6 = torch.nn.Linear(128, 64)
        self.hidden_layer7 = torch.nn.Linear(64, 32)
        self.hidden_layer8 = torch.nn.Linear(32, 16)
        self.hidden_layer9 = torch.nn.Linear(16, 8)
        self.score_predictor = torch.nn.Linear(8, 1)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        x_dict = self.conv(x_dict, edge_index_dict)

        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        combined_embedding = torch.cat((x_dict['job'], x_dict['resume']),dim=1)

        x = F.relu(self.hidden_layer1(combined_embedding))
        #x = self.dropout(x)
        x = F.relu(self.hidden_layer2(x))
        #x = self.dropout(x)
        x = F.relu(self.hidden_layer3(x))
        #x = self.dropout(x)
        x = F.relu(self.hidden_layer4(x))
        #x = self.dropout(x)
        x = F.relu(self.hidden_layer5(x))
        #x = self.dropout(x)
        x = F.relu(self.hidden_layer6(x))
        #x = self.dropout(x)
        x = F.relu(self.hidden_layer7(x))
        #x = self.dropout(x)
        x = F.relu(self.hidden_layer8(x))
        #x = self.dropout(x)
        x = F.relu(self.hidden_layer9(x))
        #x = self.dropout(x)
        score = self.score_predictor(x).squeeze()

        return score
