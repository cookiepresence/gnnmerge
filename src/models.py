import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.data import Data


class GCNBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super(GCNBackbone, self).__init__()
        if num_layers < 2:
            raise ValueError("GCNBackbone requires num_layers >= 2")
        self.num_layers = num_layers
        self.conv1: GCNConv = GCNConv(input_dim, hidden_dim)
        for i in range(2, num_layers + 1):
            setattr(self, f"conv{i}", GCNConv(hidden_dim, hidden_dim))

    def forward(self, data: Data) -> Tensor:
        x: Tensor = data.x.to(dtype=torch.float32)
        edge_index: Tensor = data.edge_index
        if hasattr(data, 'train_pos_edge_index'):
            edge_index = data.train_pos_edge_index
        for i in range(1, self.num_layers + 1):
            layer = getattr(self, f"conv{i}")
            x = layer(x, edge_index)
            x = F.relu(x)
        return x

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)


class SageBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super(SageBackbone, self).__init__()
        if num_layers < 2:
            raise ValueError("SageBackbone requires num_layers >= 2")
        self.num_layers = num_layers
        self.conv1: SAGEConv = SAGEConv(input_dim, hidden_dim)
        for i in range(2, num_layers + 1):
            setattr(self, f"conv{i}", SAGEConv(hidden_dim, hidden_dim))

    def forward(self, data: Data) -> Tensor:
        x: Tensor = data.x.to(dtype=torch.float32)
        edge_index: Tensor = data.edge_index
        if hasattr(data, 'train_pos_edge_index'):
            edge_index = data.train_pos_edge_index
        for i in range(1, self.num_layers + 1):
            layer = getattr(self, f"conv{i}")
            x = layer(x, edge_index)
            x = F.relu(x)
        return x

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)


class GATBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, heads: int = 1) -> None:
        super(GATBackbone, self).__init__()
        if num_layers < 2:
            raise ValueError("GATBackbone requires num_layers >= 2")
        self.num_layers = num_layers
        self.heads = heads
        self.conv1: GATConv = GATConv(input_dim, hidden_dim, heads=heads, concat=False)
        for i in range(2, num_layers + 1):
            setattr(self, f"conv{i}", GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))

    def forward(self, data: Data) -> Tensor:
        x: Tensor = data.x.to(dtype=torch.float32)
        edge_index: Tensor = data.edge_index
        if hasattr(data, "train_pos_edge_index"):
            edge_index = data.train_pos_edge_index
        for i in range(1, self.num_layers + 1):
            layer = getattr(self, f"conv{i}")
            x = layer(x, edge_index)
            x = F.relu(x)
        return x

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)


def _make_gin_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GINBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super(GINBackbone, self).__init__()
        if num_layers < 2:
            raise ValueError("GINBackbone requires num_layers >= 2")
        self.num_layers = num_layers
        self.conv1: GINConv = GINConv(_make_gin_mlp(input_dim, hidden_dim))
        for i in range(2, num_layers + 1):
            setattr(self, f"conv{i}", GINConv(_make_gin_mlp(hidden_dim, hidden_dim)))

    def forward(self, data: Data) -> Tensor:
        x: Tensor = data.x.to(dtype=torch.float32)
        edge_index: Tensor = data.edge_index
        if hasattr(data, "train_pos_edge_index"):
            edge_index = data.train_pos_edge_index
        for i in range(1, self.num_layers + 1):
            layer = getattr(self, f"conv{i}")
            x = layer(x, edge_index)
            x = F.relu(x)
        return x

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)


class GNNMLP(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super(GNNMLP, self).__init__()
        self.mlp: Linear = Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class GNNComplete(nn.Module):
    def __init__(self, backbone: GCNBackbone | SageBackbone | GATBackbone | GINBackbone, mlp: GNNMLP) -> None:
        super(GNNComplete, self).__init__()
        self.backbone: GCNBackbone | SageBackbone | GATBackbone | GINBackbone = backbone
        self.mlp: GNNMLP = mlp

    def forward(self, data: Data) -> Tensor:
        x: Tensor = self.backbone(data)
        return self.mlp(x)
