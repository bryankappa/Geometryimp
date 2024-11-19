import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import numpy as np

# Define number of nodes per community
num_nodes_community = 10

# Total nodes
num_nodes = 2 * num_nodes_community

# Create edges within communities
edges = []
labels = []

# Community A: nodes 0-9
for i in range(num_nodes_community):
    labels.append(0)  # Label for Community A
    for j in range(i + 1, num_nodes_community):
        edges.append([i, j])
        edges.append([j, i])  # Undirected graph

# Community B: nodes 10-19
for i in range(num_nodes_community, 2 * num_nodes_community):
    labels.append(1)  # Label for Community B
    for j in range(i + 1, 2 * num_nodes_community):
        edges.append([i, j])
        edges.append([j, i])  # Undirected graph

# Add bridge edges between communities
bridge_edges = [[num_nodes_community - 1, num_nodes_community],
                [num_nodes_community, num_nodes_community - 1]]
edges.extend(bridge_edges)
edges.extend([[edge[1], edge[0]] for edge in bridge_edges])  # Undirected

# Convert to edge index tensor
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Assign node labels
labels = torch.tensor(labels, dtype=torch.long)

# Create PyTorch Geometric Data object
data = Data(edge_index=edge_index, num_nodes=num_nodes)
data.y = labels

print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Node labels: {data.y}")

