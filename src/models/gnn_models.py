"""GNN models (GraphSAGE) using PyTorch Geometric if available."""
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    pyg_available = True
except Exception:
    pyg_available = False


class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden=128, num_layers=2, out_dim=128):
        super().__init__()
        if not pyg_available:
            raise ImportError('torch-geometric nécessaire pour GNN models')
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.pool = global_mean_pool
        self.classifier = nn.Sequential(nn.Linear(hidden, out_dim), nn.ReLU())

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        x = self.pool(x, batch)
        return self.classifier(x)
