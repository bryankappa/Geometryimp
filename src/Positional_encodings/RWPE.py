import torch
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                          [1, 0, 2, 1]], dtype=torch.long)

# features of the nodes
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

print(f"data initially: {data}")

# Add Random Walk Positional Encoding
transform = AddRandomWalkPE(walk_length= 3 )

data = transform(data)

PE = data.random_walk_pe

print(f"Random Walk Positional Encoding: {PE}")