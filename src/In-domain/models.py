import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.data import Data


class GCNBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(GCNBackbone, self).__init__()
        self.conv1: GCNConv = GCNConv(input_dim, hidden_dim)
        self.conv2: GCNConv = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data: Data) -> Tensor:
        x: Tensor = data.x.to(dtype=torch.float32)
        edge_index: Tensor = data.edge_index
        if hasattr(data, 'train_pos_edge_index'):
            edge_index = data.train_pos_edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)


class SageBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(SageBackbone, self).__init__()
        self.conv1: SAGEConv = SAGEConv(input_dim, hidden_dim)
        self.conv2: SAGEConv = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, data: Data) -> Tensor:
        x: Tensor = data.x.to(dtype=torch.float32)
        edge_index: Tensor = data.edge_index
        if hasattr(data, 'train_pos_edge_index'):
            edge_index = data.train_pos_edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
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
    def __init__(self, backbone: nn.Module, mlp: GNNMLP) -> None:
        super(GNNComplete, self).__init__()
        self.backbone: nn.Module = backbone
        self.mlp: GNNMLP = mlp

    def forward(self, data: Data) -> Tensor:
        x: Tensor = self.backbone(data)
        return self.mlp(x)
