import torch.nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from learn_pg.model import SimpleGCN
from torch_geometric.nn import GATConv, Linear, to_hetero



model = SimpleGCN(1,1,1)
hatero = to_hetero(model)
# x = torch.tensor([[1],[1]],dtype=torch.float)
# edge_index = torch.tensor([[0,1],[1,0]])
# data = Data(x=x,edge_index=edge_index)
# res = model.forward(data)
# print(res)

