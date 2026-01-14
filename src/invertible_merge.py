import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import ot  # pip install POT

import utils

# -------------------------
# Constants
# -------------------------
DEFAULT_SEED = 42
DEFAULT_MAX_ITER = 100
DEFAULT_ALPHA = 0.5

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gnn-weight-match")


# -------------------------
# Weight Matching Core
# -------------------------
class PermutationSpec:
    """Specification of which axes can be permuted in model parameters."""
    def __init__(self, perm_to_axes: dict, axes_to_perm: dict):
        self.perm_to_axes = perm_to_axes
        self.axes_to_perm = axes_to_perm

    def __str__(self):
        return f"{self.perm_to_axes=} | {self.axes_to_perm=}"


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    """Convert axes_to_perm mapping to PermutationSpec."""
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def gcn_backbone_permutation_spec() -> PermutationSpec:
    """Permutation spec for GCN backbone (2 layers)."""
    return permutation_spec_from_axes_to_perm({
        "conv1.lin.weight": ("P_0", None),
        "conv1.bias": ("P_0",),
        "conv2.lin.weight": ("P_1", "P_0"),
        "conv2.bias": ("P_1",),
    })


def gnn_complete_permutation_spec() -> PermutationSpec:
    """Permutation spec for complete GNN: backbone + MLP classifier."""
    return permutation_spec_from_axes_to_perm({
        # Backbone
        "backbone.conv1.lin.weight": ("P_0", None),
        "backbone.conv1.bias": ("P_0",),
        "backbone.conv2.lin.weight": ("P_1", "P_0"),
        "backbone.conv2.bias": ("P_1",),
        # MLP classifier (output cannot be permuted)
        "mlp.mlp.weight": (None, "P_1"),
        "mlp.mlp.bias": (None,),
    })


def sage_backbone_permutation_spec() -> PermutationSpec:
    """Permutation spec for SAGE backbone (2 layers)."""
    return permutation_spec_from_axes_to_perm({
        "conv1.lin_l.weight": ("P_0", None),
        "conv1.lin_r.weight": ("P_0", None),
        "conv1.bias": ("P_0",),
        "conv2.lin_l.weight": ("P_1", "P_0"),
        "conv2.lin_r.weight": ("P_1", "P_0"),
        "conv2.bias": ("P_1",),
    })


def extract_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract all parameters from model into flat dictionary."""
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.clone()
    for name, buffer in model.named_buffers():
        if 'running' in name or 'num_batches' in name:
            params[name] = buffer.clone()
    return params


def load_params(model: nn.Module, params: Dict[str, torch.Tensor]) -> None:
    """Load parameters from dictionary into model."""
    for name, param in model.named_parameters():
        if name in params:
            param.data.copy_(params[name])
    for name, buffer in model.named_buffers():
        if name in params:
            buffer.copy_(params[name])


def otify(cost: torch.Tensor) -> torch.Tensor:
    """
    Compute optimal transport mapping from cost matrix.
    
    Args:
        cost: Cost matrix of shape [n, n]
        
    Returns:
        Optimal transport matrix scaled by n
    """
    n = cost.shape[0]
    # Uniform distributions
    a = np.ones(n) / n
    b = np.ones(n) / n
    # Compute OT mapping
    ot_map = ot.emd(a, b, cost.cpu().numpy())
    return torch.from_numpy(ot_map).to(cost.device) * n


def build_neuron_repr_gcn(params: dict, layer_idx: int, prefix: str = "") -> torch.Tensor:
    """
    Build neuron representation for GCN layer by concatenating relevant weights.
    
    Args:
        params: Model parameters
        layer_idx: Which layer (0 for conv1, 1 for conv2)
        prefix: Prefix for parameter names (e.g., "backbone.")
        
    Returns:
        Tensor of shape [total_features, hidden_dim] representing each neuron
    """
    layer_name = f"{prefix}conv{layer_idx+1}"
    
    # Get weights for this layer
    weight = params[f"{layer_name}.lin.weight"]  # [hidden_dim, input_dim]
    bias = params[f"{layer_name}.bias"]  # [hidden_dim]
    
    # Transpose weight so neurons are columns: [input_dim, hidden_dim]
    weight_t = weight.t()
    
    # Add bias as a row: [1, hidden_dim]
    bias_row = bias.unsqueeze(0)
    
    # Concatenate incoming weights
    neuron_repr = torch.cat([weight_t, bias_row], dim=0)
    
    # If not the last layer, also include outgoing weights
    if layer_idx < 1:  # Assuming 2 layers
        next_layer_name = f"{prefix}conv{layer_idx+2}"
        next_weight = params[f"{next_layer_name}.lin.weight"]  # [next_hidden, hidden_dim]
        neuron_repr = torch.cat([neuron_repr, next_weight], dim=0)
    
    return neuron_repr


def build_neuron_repr_sage(params: dict, layer_idx: int, prefix: str = "") -> torch.Tensor:
    """
    Build neuron representation for SAGE layer by concatenating relevant weights.
    
    Args:
        params: Model parameters
        layer_idx: Which layer (0 for conv1, 1 for conv2)
        prefix: Prefix for parameter names (e.g., "backbone.")
        
    Returns:
        Tensor of shape [total_features, hidden_dim] representing each neuron
    """
    layer_name = f"{prefix}conv{layer_idx+1}"
    
    # SAGE has two weight matrices (left and right)
    weight_l = params[f"{layer_name}.lin_l.weight"]  # [hidden_dim, input_dim]
    weight_r = params[f"{layer_name}.lin_r.weight"]  # [hidden_dim, input_dim]
    bias = params[f"{layer_name}.bias"]  # [hidden_dim]
    
    # Transpose weights so neurons are columns
    weight_l_t = weight_l.t()
    weight_r_t = weight_r.t()
    
    # Add bias as a row
    bias_row = bias.unsqueeze(0)
    
    # Concatenate incoming weights
    neuron_repr = torch.cat([weight_l_t, weight_r_t, bias_row], dim=0)
    
    # If not the last layer, also include outgoing weights
    if layer_idx < 1:  # Assuming 2 layers
        next_layer_name = f"{prefix}conv{layer_idx+2}"
        next_weight_l = params[f"{next_layer_name}.lin_l.weight"]
        next_weight_r = params[f"{next_layer_name}.lin_r.weight"]
        neuron_repr = torch.cat([neuron_repr, next_weight_l, next_weight_r], dim=0)
    
    return neuron_repr


def apply_ot_transform_gcn(params: dict, layer_idx: int, P: torch.Tensor, prefix: str = "") -> dict:
    """
    Apply optimal transport transformation to GCN layer parameters.
    
    Args:
        params: Model parameters to transform
        layer_idx: Which layer (0 for conv1, 1 for conv2)
        P: Optimal transport matrix of shape [hidden_dim, hidden_dim]
        prefix: Prefix for parameter names
        
    Returns:
        Transformed parameters
    """
    layer_name = f"{prefix}conv{layer_idx+1}"
    
    # Transform output weights: W' = P @ W
    weight_key = f"{layer_name}.lin.weight"
    
    params[weight_key] = P @ params[weight_key]
    
    # Transform bias: b' = P @ b
    bias_key = f"{layer_name}.bias"
    params[bias_key] = P @ params[bias_key]
    
    # Transform input weights of next layer: W_next' = W_next @ P^T
    if layer_idx < 1:
        next_layer_name = f"{prefix}conv{layer_idx+2}"
        next_weight_key = f"{next_layer_name}.lin.weight"
        params[next_weight_key] = params[next_weight_key] @ P.t()
    
    return params


def apply_ot_transform_sage(params: dict, layer_idx: int, P: torch.Tensor, prefix: str = "") -> dict:
    """
    Apply optimal transport transformation to SAGE layer parameters.
    
    Args:
        params: Model parameters to transform
        layer_idx: Which layer (0 for conv1, 1 for conv2)
        P: Optimal transport matrix of shape [hidden_dim, hidden_dim]
        prefix: Prefix for parameter names
        
    Returns:
        Transformed parameters
    """
    layer_name = f"{prefix}conv{layer_idx+1}"
    
    # Transform output weights for both left and right
    params[f"{layer_name}.lin_l.weight"] = P @ params[f"{layer_name}.lin_l.weight"]
    params[f"{layer_name}.lin_r.weight"] = P @ params[f"{layer_name}.lin_r.weight"]
    
    # Transform bias
    params[f"{layer_name}.bias"] = P @ params[f"{layer_name}.bias"]
    
    # Transform input weights of next layer
    if layer_idx < 1:
        next_layer_name = f"{prefix}conv{layer_idx+2}"
        params[f"{next_layer_name}.lin_l.weight"] = params[f"{next_layer_name}.lin_l.weight"] @ P.t()
        params[f"{next_layer_name}.lin_r.weight"] = params[f"{next_layer_name}.lin_r.weight"] @ P.t()
    
    return params


def invertible_weight_matching(
    params_a: dict,
    params_b: dict,
    model_type: str,
    match_backbone_only: bool = False,
) -> dict:
    """
    Find optimal transport mappings to align params_b to params_a.
    
    Args:
        params_a: Reference parameters (fixed)
        params_b: Parameters to align
        model_type: Type of model ('gcn' or 'sage')
        match_backbone_only: Whether matching only backbone
        
    Returns:
        Dictionary of aligned params_b
    """
    prefix = "backbone." if not match_backbone_only else ""
    is_sage = 'sage' in model_type.lower()
    
    # Clone params_b so we can modify it
    params_b_aligned = {k: v.clone() for k, v in params_b.items()}
    
    # Match each layer
    num_layers = 2  # Assuming 2-layer GNN
    for layer_idx in range(num_layers):        
        # Build neuron representations
        if is_sage:
            repr_a = build_neuron_repr_sage(params_a, layer_idx, prefix)
            repr_b = build_neuron_repr_sage(params_b_aligned, layer_idx, prefix)
        else:
            repr_a = build_neuron_repr_gcn(params_a, layer_idx, prefix)
            repr_b = build_neuron_repr_gcn(params_b_aligned, layer_idx, prefix)
        
        # Normalize representations
        repr_a_norm = repr_a / (torch.norm(repr_a, dim=0, keepdim=True) + 1e-8)
        repr_b_norm = repr_b / (torch.norm(repr_b, dim=0, keepdim=True) + 1e-8)
        
        # Compute cost matrix (L1 distance between normalized representations)
        cost = torch.cdist(repr_a_norm.t(), repr_b_norm.t(), p=1)
        
        logger.info(f"  Cost matrix shape: {cost.shape}, mean cost: {cost.mean().item():.4f}")
        
        # Compute optimal transport mapping
        P = otify(cost).to(torch.float)
        
        logger.info(f"  OT matrix shape: {P.shape}, sparsity: {(P > 1e-6).float().mean().item():.4f}")
        
        # Apply transformation to align params_b
        if is_sage:
            params_b_aligned = apply_ot_transform_sage(params_b_aligned, layer_idx, P.t(), prefix)
        else:
            params_b_aligned = apply_ot_transform_gcn(params_b_aligned, layer_idx, P.t(), prefix)
    
    logger.info("Invertible weight matching complete")
    return params_b_aligned


def interpolate_params(params_a: dict, params_b: dict, alpha: float) -> dict:
    """Linearly interpolate between two parameter dictionaries."""
    return {
        k: alpha * params_a[k] + (1 - alpha) * params_b[k]
        for k in params_a.keys()
    }


# -------------------------
# Dataset Loading
# -------------------------
def load_dataset(path: Path):
    """Load dataset and return dataset object with metadata."""
    ds = torch.load(str(path), map_location=utils.get_device(), weights_only=False)
    num_nodes = int(ds.num_nodes)
    num_labels = int(len(ds.label_names))
    input_dim = int(ds.x.size(1))
    logger.info("Loaded dataset from %s", path)
    logger.info("Nodes=%d; Labels=%d; InputDim=%d", num_nodes, num_labels, input_dim)
    return ds, num_nodes, num_labels, input_dim


MaskType = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
ClassType = tuple[set[int], set[int]]


def make_label_masks(dataset, num_nodes: int, num_labels: int) -> tuple[MaskType, ClassType]:
    """
    Create boolean masks for train/val/test for both label halves,
    and remap labels for the second model (so they are 0..C2-1).
    """
    classes1 = set(range(0, (num_labels + 1) // 2))
    classes2 = set(range((num_labels + 1) // 2, num_labels))

    device = torch.device("cpu")

    mk = lambda: torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask1, train_mask2 = mk(), mk()
    val_mask1, val_mask2 = mk(), mk()
    test_mask1, test_mask2 = mk(), mk()

    train_indices = dataset.train_masks[0]
    val_indices = dataset.val_masks[0]
    test_indices = dataset.test_masks[0]

    for idx in train_indices:
        label = int(dataset.y[idx].item())
        (train_mask1 if label in classes1 else train_mask2)[idx] = True

    for idx in val_indices:
        label = int(dataset.y[idx].item())
        (val_mask1 if label in classes1 else val_mask2)[idx] = True

    for idx in test_indices:
        label = int(dataset.y[idx].item())
        (test_mask1 if label in classes1 else test_mask2)[idx] = True

    label_mapping = {old: new for new, old in enumerate(sorted(list(classes2)))}
    y_copy = dataset.y.clone()
    for i in range(len(y_copy)):
        if test_mask2[i] or train_mask2[i] or val_mask2[i]:
            y_copy[i] = label_mapping[int(y_copy[i].item())]
    dataset.y = y_copy

    return (train_mask1, train_mask2, val_mask1, val_mask2, test_mask1, test_mask2), (classes1, classes2)


# -------------------------
# Evaluation
# -------------------------
def evaluate_model(model: nn.Module, dataset, mask_tuple) -> tuple:
    """Evaluate model on train/val/test splits."""
    model.eval()
    with torch.no_grad():
        out = model(dataset)
        pred = out.argmax(dim=1)

        train_mask, val_mask, test_mask = mask_tuple
        train_acc = (pred[train_mask] == dataset.y[train_mask]).float().mean()
        val_acc = (pred[val_mask] == dataset.y[val_mask]).float().mean()
        test_acc = (pred[test_mask] == dataset.y[test_mask]).float().mean()

    return train_acc.item(), val_acc.item(), test_acc.item()


# -------------------------
# Save/Load
# -------------------------
def save(
    path: Path,
    merged_model: Optional[nn.Module] = None,
    models: Optional[list[nn.Module]] = None,
    metadata: Optional[dict[str, Any]] = None,
    logs: Optional[dict] = None
):
    """Save models, metadata, and logs."""
    path.mkdir(parents=True, exist_ok=True)

    if merged_model is not None:
        torch.save(merged_model.state_dict(), path / "model.pt")
        logger.info("Saved merged model to %s", path / "model.pt")

    if models is not None:
        for i, model in enumerate(models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")

    if metadata is not None:
        metadata_file = path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

    if logs is not None:
        log_file = path / "logs.json"
        log_file.write_text(json.dumps(logs, indent=2))


# -------------------------
# Main Matching Function
# -------------------------
def match_models(
    models: list[tuple[nn.Module, dict]],
    datasets: dict,
    save_path: Path,
    alpha: float,
    seed: int,
    match_backbone_only: bool = False,
):
    """
    Perform invertible weight matching on a list of models.
    """
    if len(models) != 2:
        raise ValueError("Weight matching currently supports exactly 2 models")

    (model_a, meta_a), (model_b, meta_b) = models

    logger.info("=" * 80)
    logger.info("Model A: %s (dataset=%s)", meta_a.get('model_type', 'unknown'), meta_a['dataset'])
    logger.info("Model B: %s (dataset=%s)", meta_b.get('model_type', 'unknown'), meta_b['dataset'])
    logger.info("=" * 80)

    model_type = meta_a['model_type']
    
    # Extract the backbone if needed
    if match_backbone_only:
        model_a = model_a.backbone
        model_b = model_b.backbone
        logger.info("Matching backbone only")
    else:
        logger.info("Matching complete model (backbone + MLP)")

    # Extract parameters
    logger.info("Extracting parameters...")
    params_a = extract_params(model_a)
    params_b = extract_params(model_b)

    logger.info("Model A parameters: %s", list(params_a.keys()))
    logger.info("Model B parameters: %s", list(params_b.keys()))

    # Run invertible weight matching
    logger.info("Running invertible weight matching with optimal transport...")
    params_b_aligned = params_b
    for _ in range(20):
        params_b_aligned = invertible_weight_matching(
            params_a, params_b_aligned, model_type, match_backbone_only
        )

    # Create interpolated model
    logger.info("Creating interpolated model (alpha=%.2f)...", alpha)
    params_interp = interpolate_params(params_a, params_b_aligned, alpha)

    # Build merged model
    if not match_backbone_only:
        merged_model = utils.build_model(
            model_name=model_type,
            input_dim=meta_a['input_dim'],
            num_labels=meta_a['num_labels'],
            device=utils.get_device(),
            hidden_dim=meta_a['hidden_dim']
        )
        load_params(merged_model, params_interp)
    else:
        merged_model = model_a.__class__(
            input_dim=meta_a['input_dim'],
            hidden_dim=meta_a['hidden_dim']
        ).to(utils.get_device())
        load_params(merged_model, params_interp)

    # Prepare metadata
    metadata = {
        "method": "invertible_weight_matching",
        "alpha": alpha,
        "seed": seed,
        "backbone_only": match_backbone_only,
        "model_type": model_type,
        "source_models": [meta_a, meta_b],
    }

    # Evaluate if we have a complete model
    logs = {}
    if not match_backbone_only and len(datasets) > 0:
        logger.info("Evaluating merged model...")
        dataset_name = meta_a['dataset']
        if dataset_name in datasets:
            ds, num_nodes, num_labels, input_dim = datasets[dataset_name]
            masks, classes = make_label_masks(ds, num_nodes, num_labels)
            
            m1 = (masks[0], masks[2], masks[4])
            train_acc, val_acc, test_acc = evaluate_model(merged_model, ds, m1)
            logger.info("Merged model accuracy (split 1):")
            logger.info("  Train: %.4f", train_acc)
            logger.info("  Val:   %.4f", val_acc)
            logger.info("  Test:  %.4f", test_acc)

            m2 = (masks[1], masks[3], masks[5])
            train_acc, val_acc, test_acc = evaluate_model(merged_model, ds, m2)
            logger.info("Merged model accuracy (split 2):")
            logger.info("  Train: %.4f", train_acc)
            logger.info("  Val:   %.4f", val_acc)
            logger.info("  Test:  %.4f", test_acc)

            logs['merged_model'] = {
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc
            }

    # Save everything
    logger.info("Saving to %s...", save_path)
    save(
        save_path,
        merged_model=merged_model,
        models=[model_a, model_b] if not match_backbone_only else None,
        metadata=metadata,
        logs=logs if logs else None
    )

    logger.info("=" * 80)
    logger.info("Invertible weight matching complete!")
    logger.info("=" * 80)


# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Invertible weight matching for GNN models using optimal transport"
    )
    parser.add_argument(
        "--model-path",
        action="append",
        type=Path,
        required=True,
        help="Path to model (use twice for two models)"
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        required=True,
        help="Where to save merged model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Interpolation coefficient (0=first model, 1=second model)"
    )
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="Match only backbone (not full model)"
    )

    args = parser.parse_args()

    if args.model_path is None or len(args.model_path) != 2:
        logger.error("Must provide exactly 2 model paths")
        parser.print_help()
        exit(1)

    # Set seed
    utils.__init__randomness__(args.seed)

    device = utils.get_device()

    # Load models
    logger.info("Loading models...")
    models = [
        utils.load_models(path, task=utils.Task.NodeClassification, device=device)
        for path in args.model_path
    ]

    # Load datasets
    dataset_names = set([metadata['dataset'] for _, metadata in models])
    logger.info("Loading datasets: %s", dataset_names)
    datasets = {
        ds: load_dataset(Path('artifacts/datasets') / (ds + ".pt"))
        for ds in dataset_names
    }

    # Run weight matching
    match_models(
        models=models,
        datasets=datasets,
        save_path=args.save_path,
        alpha=args.alpha,
        seed=args.seed,
        match_backbone_only=args.backbone_only
    )
