import pathlib
import enum
import json
from typing import Literal, Optional

import random
import numpy as np
import torch
import torch_geometric

import models as models

class Task(enum.StrEnum):
    NodeClassification = 'Node Classification'
    LinkPrediction = 'Link Prediction'

def build_model(
        model_name: Literal['gcn', 'sage'],
        input_dim: int,
        num_labels: Optional[int],
        device: torch.device,
        hidden_dim: int
) -> torch.nn.Module:
    if model_name == 'gcn':
        backbone = models.GCNBackbone(input_dim, hidden_dim).to(device)
    elif model_name == 'sage':
        backbone = models.SageBackbone(input_dim, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name!r}")

    # if we are not doing node classification -- return only backbone
    if num_labels is None:
        return backbone

    mlp = models.GNNMLP(hidden_dim, num_labels).to(device)
    model = models.GNNComplete(backbone, mlp).to(device)

    return model


def load_models(path: pathlib.Path, task: Task, device: torch.device) -> tuple[torch.nn.Module, dict]:
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)

    model = build_model(
        model_name=metadata["model_type"],
        input_dim=metadata["input_dim"],
        hidden_dim=metadata["hidden_dim"],
        num_labels=metadata["num_labels"],
        device=device,
    )

    state_dict = torch.load(path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)

    return model, metadata


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
