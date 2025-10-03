import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub
from torch_geometric.data import Data

# Import dataset from PyTorch Geometric
dataset = KarateClub()
data = dataset[0]

# edge_index = torch.tensor([[0, 1],
#                            [1, 0]], dtype=torch.long)
# x = torch.tensor([[1], [2]], dtype=torch.float)
#
# my_test = Data(x=x,edge_index=edge_index)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(dataset.num_features, 3)
        self.out = Linear(3, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        return h, z
        #return  h

model = GCN()
print(model)

h = model(data.x, data.edge_index)
print(h)

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
#
# # Calculate accuracy
# def accuracy(pred_y, y):
#     return (pred_y == y).sum() / len(y)
#
# # Data for animations
# embeddings = []
# losses = []
# accuracies = []
# outputs = []
#
# # Training loop
# for epoch in range(201):
#     # Clear gradients
#     optimizer.zero_grad()
#
#     # Forward pass
#     h, z = model(data.x, data.edge_index)
#
#     # Calculate loss function
#     loss = criterion(z, data.y)
#
#     # Calculate accuracy
#     acc = accuracy(z.argmax(dim=1), data.y)
#
#     # Compute gradients
#     loss.backward()
#
#     # Tune parameters
#     optimizer.step()
#
#     # Store data for animations
#     embeddings.append(h)
#     losses.append(loss)
#     accuracies.append(acc)
#     outputs.append(z.argmax(dim=1))
#
#     # Print metrics every 10 epochs
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')