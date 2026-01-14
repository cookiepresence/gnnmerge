import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

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


def get_permuted_param(
    ps: PermutationSpec,
    perm: dict,
    k: str,
    params: dict,
    except_axis: Optional[int] = None
) -> torch.Tensor:
    """Get parameter k with permutations applied (except for except_axis)."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue
        if p is not None:
            w = torch.index_select(w, axis, perm[p].long().to(utils.get_device()))
    return w


def apply_permutation(ps: PermutationSpec, perm: dict, params: dict) -> dict:
    """Apply permutation to all parameters."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(
    ps: PermutationSpec,
    params_a: dict,
    params_b: dict,
    max_iter: int = 100,
    init_perm: Optional[dict] = None
) -> dict:
    """
    Find optimal permutation to align params_b to params_a.

    Args:
        ps: PermutationSpec defining the structure
        params_a: Reference parameters (fixed)
        params_b: Parameters to permute
        max_iter: Maximum iterations
        init_perm: Initial permutation (optional)

    Returns:
        Dictionary mapping permutation names to permutation tensors
    """
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]]
        for p, axes in ps.perm_to_axes.items()
    }

    perm = (
        {p: torch.arange(n) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )
    perm_names = list(perm.keys())

    logger.info("Starting weight matching with %d permutation groups", len(perm_names))

    for iteration in range(max_iter):
        progress = False
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n))

            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).cpu()
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).cpu()
                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A.cpu().detach().numpy(), maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()

            oldL = torch.einsum('ij,ij->i', A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum('ij,ij->i', A, torch.eye(n)[ci, :]).sum()

            improvement = newL - oldL
            if improvement > 1e-12:
                logger.debug("Iter %d/%s: improvement=%.6f", iteration, p, improvement.item())
                progress = True
                perm[p] = torch.tensor(ci, dtype=torch.float32)

        if not progress:
            logger.info("Converged after %d iterations", iteration + 1)
            break
    else:
        logger.info("Reached max iterations (%d)", max_iter)

    return perm


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

# TODO: move to utils!!!
MaskType = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
ClassType = tuple[set[int], set[int]]
def make_label_masks(dataset, num_nodes: int, num_labels: int) -> tuple[MaskType, ClassType]:
    """
    Create boolean masks for train/val/test for both label halves,
    and remap labels for the second model (so they are 0..C2-1).
    Returns tuple:
      (train_mask1, train_mask2, val_mask1, val_mask2, test_mask1, test_mask2, classes_set1, classes_set2)
    """
    # class splits
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

    # set masks
    for idx in train_indices:
        label = int(dataset.y[idx].item())
        (train_mask1 if label in classes1 else train_mask2)[idx] = True

    for idx in val_indices:
        label = int(dataset.y[idx].item())
        (val_mask1 if label in classes1 else val_mask2)[idx] = True

    for idx in test_indices:
        label = int(dataset.y[idx].item())
        (test_mask1 if label in classes1 else test_mask2)[idx] = True

    # remap labels for model 2 to range 0..C2-1
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
    permutations: Optional[dict] = None,
    metadata: Optional[dict[str, Any]] = None,
    logs: Optional[dict] = None
):
    """Save models, permutations, metadata, and logs."""
    path.mkdir(parents=True, exist_ok=True)

    if merged_model is not None:
        torch.save(merged_model.state_dict(), path / "model.pt")
        logger.info("Saved merged model to %s", path / "model.pt")

    if models is not None:
        for i, model in enumerate(models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")

    if permutations is not None:
        perm_serializable = {k: v.tolist() for k, v in permutations.items()}
        perm_file = path / "permutations.json"
        perm_file.write_text(json.dumps(perm_serializable, indent=2))
        logger.info("Saved permutations to %s", perm_file)

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
    max_iter: int,
    alpha: float,
    seed: int,
    match_backbone_only: bool = False,
):
    """
    Perform weight matching on a list of models.

    Args:
        models: List of (model, metadata) tuples
        datasets: Dictionary of datasets
        save_path: Where to save results
        max_iter: Maximum iterations for weight matching
        alpha: Interpolation coefficient (0=model_a, 1=model_b)
        seed: Random seed
        match_backbone_only: If True, only match backbone (not full model)
    """
    if len(models) != 2:
        raise ValueError("Weight matching currently supports exactly 2 models")

    (model_a, meta_a), (model_b, meta_b) = models

    logger.info("=" * 80)
    logger.info("Model A: %s (dataset=%s)", meta_a.get('model_type', 'unknown'), meta_a['dataset'])
    logger.info("Model B: %s (dataset=%s)", meta_b.get('model_type', 'unknown'), meta_b['dataset'])
    logger.info("=" * 80)

    # Determine permutation spec based on architecture
    model_type = meta_a['model_type']
    if match_backbone_only:
        if 'gcn' in model_type.lower():
            ps = gcn_backbone_permutation_spec()
            model_a = model_a.backbone
            model_b = model_b.backbone
        elif 'sage' in model_type.lower():
            ps = sage_backbone_permutation_spec()
            model_a = model_a.backbone
            model_b = model_b.backbone
        else:
            raise ValueError(f"Unknown model type for backbone: {model_type}")
        logger.info("Matching backbone only")
    else:
        ps = gnn_complete_permutation_spec()
        logger.info("Matching complete model (backbone + MLP)")

    # Extract parameters
    logger.info("Extracting parameters...")
    params_a = extract_params(model_a)
    params_b = extract_params(model_b)

    logger.info("Model A parameters: %s", list(params_a.keys()))
    logger.info("Model B parameters: %s", list(params_b.keys()))

    # Run weight matching
    logger.info("Running weight matching (max_iter=%d)...", max_iter)
    perm = weight_matching(ps, params_a, params_b, max_iter=max_iter)

    logger.info("Found permutations: %s", {k: v.shape for k, v in perm.items()})

    # Apply permutation to model_b
    logger.info("Applying permutation to Model B...")
    params_b_aligned = apply_permutation(ps, perm, params_b)

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
        # For backbone-only, create just the backbone
        merged_model = model_a.__class__(
            input_dim=meta_a['input_dim'],
            hidden_dim=meta_a['hidden_dim']
        ).to(utils.get_device())
        load_params(merged_model, params_interp)

    # Prepare metadata
    metadata = {
        "method": "weight_matching",
        "max_iter": max_iter,
        "alpha": alpha,
        "seed": seed,
        "backbone_only": match_backbone_only,
        "model_type": model_type,
        "source_models": [meta_a, meta_b],
        "permutation_shapes": {k: list(v.shape) for k, v in perm.items()},
    }

    # Evaluate if we have a complete model
    logs = {}
    if not match_backbone_only and len(datasets) > 0:
        logger.info("Evaluating merged model...")
        dataset_name = meta_a['dataset']
        if dataset_name in datasets:
            ds, num_nodes, num_labels, input_dim = datasets[dataset_name]
            masks, classes = make_label_masks(ds, num_nodes, num_labels)
            # Use first train/val/test mask
            m1 = (masks[0], masks[2], masks[4])
            train_acc, val_acc, test_acc = evaluate_model(merged_model, ds, m1)

            logger.info("Merged model accuracy:")
            logger.info("  Train: %.4f", train_acc)
            logger.info("  Val:   %.4f", val_acc)
            logger.info("  Test:  %.4f", test_acc)


            m2 = (masks[1], masks[3], masks[5])
            train_acc, val_acc, test_acc = evaluate_model(merged_model, ds, m2)

            logger.info("Merged model accuracy:")
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
        permutations=perm,
        metadata=metadata,
        logs=logs if logs else None
    )

    logger.info("=" * 80)
    logger.info("Weight matching complete!")
    logger.info("=" * 80)


# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Weight matching for GNN models"
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
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Maximum iterations for weight matching"
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
        max_iter=args.max_iter,
        alpha=args.alpha,
        seed=args.seed,
        match_backbone_only=args.backbone_only
    )
