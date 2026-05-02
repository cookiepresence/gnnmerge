from pathlib import Path
import enum
import json
from typing import Any, Literal, Optional

import random
import numpy as np
import torch

import models as models

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def __init__randomness__(seed):
    import os

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Task(enum.StrEnum):
    NodeClassification = 'Node Classification'
    LinkPrediction = 'Link Prediction'

def build_model(
        model_name: Literal['gcn', 'sage', 'gat', 'gin'],
        input_dim: int,
        num_labels: Optional[int],
        device: torch.device,
        hidden_dim: int,
        num_layers: int = 2,
) -> torch.nn.Module:
    if model_name == 'gcn':
        backbone = models.GCNBackbone(input_dim, hidden_dim, num_layers=num_layers).to(device)
    elif model_name == 'sage':
        backbone = models.SageBackbone(input_dim, hidden_dim, num_layers=num_layers).to(device)
    elif model_name == 'gat':
        backbone = models.GATBackbone(input_dim, hidden_dim, num_layers=num_layers).to(device)
    elif model_name == 'gin':
        backbone = models.GINBackbone(input_dim, hidden_dim, num_layers=num_layers).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name!r}")

    # if we are not doing node classification -- return only backbone
    if num_labels is None:
        return backbone

    mlp = models.GNNMLP(hidden_dim, num_labels).to(device)
    model = models.GNNComplete(backbone, mlp).to(device)

    return model


def load_model_raw(weight: Path, metadata: dict, device: torch.device):
    model = build_model(
        model_name=metadata["model_type"],
        input_dim=metadata["input_dim"],
        hidden_dim=metadata["hidden_dim"],
        num_layers=metadata.get("num_layers", 2),
        num_labels=metadata.get("num_labels", None),
        device=device,
    )

    state_dict = torch.load(weight, map_location=device)
    model.load_state_dict(state_dict)

    return model


def load_models(path: Path, task: Task, device: torch.device) -> tuple[torch.nn.Module, dict]:
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)

    model = load_model_raw(path / "model.pt", metadata, device)

    return model, metadata


def load_dataset(path: Path):
    """
    Load dataset with torch.load (map to CPU) and return dataset and metadata.
    Note: dataset is returned on CPU. Caller may move it to `device`.
    """
    ds = torch.load(str(path), map_location="cpu", weights_only=False)
    num_nodes = int(ds.num_nodes)
    num_labels = int(len(ds.label_names))
    input_dim = int(ds.x.size(1))
    return ds, num_nodes, num_labels, input_dim


def graph_size(data) -> tuple[int, int]:
    num_nodes = int(getattr(data, "num_nodes", 0))
    if hasattr(data, "num_edges") and data.num_edges is not None:
        num_edges = int(data.num_edges)
    elif hasattr(data, "edge_index") and data.edge_index is not None:
        num_edges = int(data.edge_index.size(1))
    else:
        num_edges = 0
    return num_nodes, num_edges



def make_inductive_subgraph(data, mask):
    train_mask, val_mask, test_mask = mask
    subset = (train_mask | val_mask | test_mask).nonzero(as_tuple=False).view(-1)

    if hasattr(data, 'adj_t') and data.adj_t is not None:
        data = copy.copy(data)
        # adj_t is the transposed adjacency: its COO (row, col) = (dst, src)
        # so flip to get (src, dst) for edge_index
        dst, src, val = data.adj_t.coo()
        data.edge_index = torch.stack([src, dst], dim=0)
        if val is not None:
            data.edge_attr = val
        del data.adj_t

    subgraph_data = data.subgraph(subset)
    subgraph_mask = (
        train_mask[subset],
        val_mask[subset],
        test_mask[subset],
    )
    return subgraph_data, subgraph_mask

def resolve_split_mode(source_metadata: list[dict[str, Any]]) -> str:
    known_modes = {
        str(meta.get("split_mode"))
        for meta in source_metadata
        if meta.get("split_mode") in {"inductive", "transductive"}
    }
    if len(known_modes) == 1:
        return next(iter(known_modes))
    if len(known_modes) > 1:
        raise ValueError(f"Inconsistent split_mode across source checkpoints: {sorted(known_modes)}")

    # Backward-compat fallback for older checkpoints that did not store split_mode.
    return "transductive"


def save(
    path: Path,
    merged_model: Optional[torch.nn.Module] = None,
    models: Optional[list[torch.nn.Module]] = None,
    metadata: Optional[dict[str, Any]] = None,
    logs: Optional[dict] = None,
    aux_state: Optional[dict[str, dict[str, torch.Tensor]]] = None,
):
    path.mkdir(parents=True, exist_ok=True)
    if merged_model is not None:
        torch.save(merged_model.state_dict(), path / "model.pt")
    if models is not None:
        for i, model in enumerate(models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")
    if aux_state is not None:
        for name, state_dict in aux_state.items():
            torch.save(state_dict, path / f"{name}.pt")
    if metadata is not None:
        metadata_file = path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
    if logs is not None:
        log_file = path / "logs.json"
        log_file.write_text(json.dumps(logs, indent=2))

def labels_in_class(class_idx: int, num_labels: int, num_classes: int):
    start = (class_idx * num_labels) // num_classes
    end = ((class_idx + 1) * num_labels) // num_classes
    return end - start

MaskType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

def make_label_masks(
    dataset,
    num_nodes: int,
    num_labels: int,
    num_classes: int = 2
) -> list[MaskType]:
    """
    Create boolean masks for train/val/test splits across multiple class groups.

    Args:
        dataset: Dataset with train_masks, val_masks, test_masks, and y attributes
        num_nodes: Total number of nodes in the graph
        num_labels: Total number of unique labels
        num_classes: Number of class groups to split labels into (default: 2)
    """
    device = get_device()

    # Get indices for each split
    train_indices = dataset.train_masks[0]
    val_indices = dataset.val_masks[0]
    test_indices = dataset.test_masks[0]

    # Special case: num_classes == 1, no splitting needed
    if num_classes == 1:
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        return [(train_mask, val_mask, test_mask)]

    # Get labels for each split (vectorized)
    train_labels = dataset.y[train_indices].long()
    val_labels = dataset.y[val_indices].long()
    test_labels = dataset.y[test_indices].long()

    # Create a lookup tensor that maps each label to its class
    label_to_class = torch.zeros(num_labels, dtype=torch.long, device=device)
    for class_idx in range(num_classes):
        start = (class_idx * num_labels) // num_classes
        end = ((class_idx + 1) * num_labels) // num_classes
        label_to_class[start:end] = class_idx

    # Compute which class group each label belongs to (vectorized lookup)
    train_class_ids = label_to_class[train_labels]
    val_class_ids = label_to_class[val_labels]
    test_class_ids = label_to_class[test_labels]

    # Initialize all masks
    masks = []

    for class_idx in range(num_classes):
        # Vectorized membership check
        train_in_class = train_class_ids == class_idx
        val_in_class = val_class_ids == class_idx
        test_in_class = test_class_ids == class_idx

        # Create empty masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # Set True values in a single operation per mask
        train_mask[train_indices[train_in_class]] = True
        val_mask[val_indices[val_in_class]] = True
        test_mask[test_indices[test_in_class]] = True

        masks.append((train_mask, val_mask, test_mask))

    # Remap labels (vectorized)
    y_remapped = dataset.y.clone()
    all_indices = torch.cat([train_indices, val_indices, test_indices])
    all_labels = dataset.y[all_indices].long()
    all_class_ids = label_to_class[all_labels]

    for class_idx in range(1, num_classes):
        in_class = all_class_ids == class_idx
        nodes_to_remap = all_indices[in_class]

        class_start = (class_idx * num_labels) // num_classes
        y_remapped[nodes_to_remap] = dataset.y[nodes_to_remap].long() - class_start

    dataset.y = y_remapped

    return masks
