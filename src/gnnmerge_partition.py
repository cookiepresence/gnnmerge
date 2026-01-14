#!/usr/bin/env python3
"""
Train a partition layer to decompose merged model embeddings.

The partition layer learns a soft binary mask that satisfies:
    Embed(M(x)) * l ≈ Embed(M_1(x))
    Embed(M(x)) * (1 - l) ≈ Embed(M_2(x))

Usage:
    python train_partition_layer.py \
        --merged-model-path /path/to/merged/model \
        --model1-path /path/to/model1 \
        --model2-path /path/to/model2 \
        --data-path /path/to/dataset.pt \
        --save-path /path/to/partition_layer.pt \
        --num-epochs 1000
"""

import argparse
import collections
import json
import logging
import time
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


MaskType = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
ClassType = tuple[set[int], set[int]]

# -------------------------
# Configurable hyperparams
# -------------------------
DEFAULT_SEED = 42
DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-4
DEFAULT_EPOCHS = 1000
DEFAULT_TEMP = 1.0  # Temperature for sigmoid (lower = more binary)

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("partition-layer-train")


class PartitionLayer(nn.Module):
    """
    Learnable soft binary mask for partitioning embeddings.
    
    Args:
        hidden_dim: Dimension of the embeddings
        temperature: Temperature parameter for sigmoid (lower = more binary)
    """
    def __init__(self, hidden_dim: int, temperature: float = DEFAULT_TEMP):
        super().__init__()
        self.temperature = temperature
        # Initialize logits near 0 (sigmoid(0) = 0.5)
        self.logits = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mask: Soft binary mask in [0, 1]
            complementary_mask: 1 - mask
        """
        mask = torch.sigmoid(self.logits / self.temperature)
        return mask, 1 - mask
    
    def get_binary_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Get hard binary mask for evaluation."""
        with torch.no_grad():
            mask = torch.sigmoid(self.logits / self.temperature)
            return (mask > threshold).float()


def load_dataset(path: Path):
    """Load dataset and return dataset + metadata."""
    ds = torch.load(str(path), map_location="cpu", weights_only=False)
    num_nodes = int(ds.num_nodes)
    num_labels = int(len(ds.label_names))
    input_dim = int(ds.x.size(1))
    logger.info("Loaded dataset from %s", path)
    logger.info("Nodes=%d; Labels=%d; InputDim=%d", num_nodes, num_labels, input_dim)
    return ds, num_nodes, num_labels, input_dim


def save(
    path: Path,
    partition_layer: Optional[nn.Module] = None,
    metadata: Optional[dict[str, Any]] = None,
    logs: Optional[dict] = None
):
    """Save partition layer, metadata, and training logs."""
    path.mkdir(parents=True, exist_ok=True)
    if partition_layer is not None:
        torch.save(partition_layer.state_dict(), path / "partition_layer.pt")
    if metadata is not None:
        metadata_file = path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
    if logs is not None:
        log_file = path / "logs.json"
        log_file.write_text(json.dumps(logs, indent=2))


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

        
def compute_embeddings(model: nn.Module, data) -> torch.Tensor:
    """Extract embeddings from model's backbone."""
    model.eval()
    with torch.no_grad():
        # Assuming model has a backbone attribute that produces embeddings
        if hasattr(model, 'backbone'):
            embeddings = model.backbone(data)
        else:
            # Fallback: use the full model output before classification
            embeddings = model(data)
        return embeddings


def train_step(
    partition_layer: nn.Module,
    merged_embeddings: torch.Tensor,
    model1_embeddings: torch.Tensor,
    model2_embeddings: torch.Tensor,
    optimizer,
    criterion,
    sparsity_weight: float = 0.0
) -> tuple[float, float, float, float]:
    """
    Train one step of the partition layer.
    
    Returns:
        total_loss, model1_loss, model2_loss, sparsity_loss
    """
    partition_layer.train()
    optimizer.zero_grad()
    
    # Get soft masks
    mask1, mask2 = partition_layer(merged_embeddings)
    
    # Apply masks to merged embeddings
    partitioned1 = merged_embeddings * mask1.unsqueeze(0)
    partitioned2 = merged_embeddings * mask2.unsqueeze(0)
    
    # Reconstruction losses
    loss1 = criterion(partitioned1, model1_embeddings)
    loss2 = criterion(partitioned2, model2_embeddings)
    
    # Optional: Sparsity regularization to encourage binary-like masks
    # This encourages mask values to be close to 0 or 1
    sparsity_loss = torch.tensor(0.0, device=merged_embeddings.device)
    if sparsity_weight > 0:
        mask_probs = torch.sigmoid(partition_layer.logits / partition_layer.temperature)
        # Entropy-based sparsity: penalize uncertainty (values near 0.5)
        entropy = -mask_probs * torch.log(mask_probs + 1e-8) - (1 - mask_probs) * torch.log(1 - mask_probs + 1e-8)
        sparsity_loss = entropy.mean()
    
    total_loss = loss1 + loss2 + sparsity_weight * sparsity_loss
    total_loss.backward()
    optimizer.step()
    
    return (
        float(total_loss.item()),
        float(loss1.item()),
        float(loss2.item()),
        float(sparsity_loss.item())
    )


def evaluate_partition(
    partition_layer: PartitionLayer,
    merged_model: nn.Module,
    models: list[nn.Module],
    datasets: dict[str, Any],
    model_metadata: list[dict],
    masks: tuple,
    criterion
) -> tuple[float, float, float, float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate partition layer with both reconstruction loss and classification accuracy.
    
    Returns:
        total_loss, model1_loss, model2_loss, mask_mean, mask_std, train_accs, val_accs, test_accs
    """
    partition_layer.eval()
    merged_model.eval()
    for model in models:
        model.eval()
    
    with torch.no_grad():
        # Get embeddings for loss computation
        merged_embeddings = merged_model(datasets[model_metadata[0]['dataset']])
        model1_embeddings = models[0].backbone(datasets[model_metadata[0]['dataset']])
        model2_embeddings = models[1].backbone(datasets[model_metadata[1]['dataset']])
        
        # Compute reconstruction losses
        mask1, mask2 = partition_layer(merged_embeddings)
        partitioned1 = merged_embeddings * mask1.unsqueeze(0)
        partitioned2 = merged_embeddings * mask2.unsqueeze(0)
        
        loss1 = criterion(partitioned1, model1_embeddings)
        loss2 = criterion(partitioned2, model2_embeddings)
        total_loss = loss1 + loss2
        
        # Compute mask statistics
        mask_probs = torch.sigmoid(partition_layer.logits / partition_layer.temperature)
        mask_mean = float(mask_probs.mean().item())
        mask_std = float(mask_probs.std().item())
        
        # Compute classification accuracies
        # Get full model outputs (including MLP head)
        merged_outputs = [merged_model(datasets[meta['dataset']]) for meta in model_metadata]
        merged_outputs[0] = merged_outputs[0] * mask1.unsqueeze(0)
        merged_outputs[1] = merged_outputs[1] * mask2.unsqueeze(0)
        
        # Get predictions from each model's MLP head applied to merged embeddings
        predictions = [model.mlp(out).argmax(dim=1) for model, out in zip(models, merged_outputs)]
        predictions = torch.stack(predictions)
        labels = torch.stack([datasets[meta['dataset']].y for meta in model_metadata])
        
        # predictions shape: [2, num_nodes]
        # labels shape: [2, num_nodes]
        # masks shape: [6, num_nodes]
        
        # Stack the masks so that we can index them comfortably
        masks_tensor = torch.stack(masks).to(merged_embeddings.device)
        masks_tensor = masks_tensor.view(3, len(models), -1)
        
        # Compute correct predictions: [num_models, num_samples]
        correct = (predictions == labels).float()
        
        # Expand dimensions for broadcasting: [1, num_models, num_samples]
        correct = correct.unsqueeze(0)
        
        # Apply masks and compute accuracies: [3, num_models]
        # Sum correct predictions in each mask, divide by mask size
        masked_correct = (correct * masks_tensor).sum(dim=2)  # [3, num_models]
        mask_sizes = masks_tensor.sum(dim=2)  # [3, num_models]
        
        # Avoid division by zero
        accuracies = masked_correct / mask_sizes.clamp(min=1)
        
        # Split into train, val, test: each of shape [num_models]
        train_accs = accuracies[0]
        val_accs = accuracies[1]
        test_accs = accuracies[2]
        
        return (
            float(total_loss.item()),
            float(loss1.item()),
            float(loss2.item()),
            mask_mean,
            mask_std,
            train_accs,
            val_accs,
            test_accs
        )


def train_partition_layer(
    partition_layer: PartitionLayer,
    merged_model: nn.Module,
    model1: tuple[nn.Module, dict],
    model2: tuple[nn.Module, dict],
    datasets: dict[str, Any],
    masks: tuple,
    save_path: Path,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    sparsity_weight: float,
    temperature: float,
    seed: int
):
    """Train the partition layer to decompose merged model embeddings."""
    
    # Unpack models and metadata
    model1_net, model1_metadata = model1
    model2_net, model2_metadata = model2
    models = [model1_net, model2_net]
    source_metadata = [model1_metadata, model2_metadata]
    
    # Get dataset names from metadata
    dataset1_name = model1_metadata['dataset']
    dataset2_name = model2_metadata['dataset']
    
    # Extract embeddings once (models are frozen) for training
    logger.info("Extracting embeddings from models...")
    logger.info("Model 1 uses dataset: %s", dataset1_name)
    logger.info("Model 2 uses dataset: %s", dataset2_name)
    
    # Note: merged model should work on both datasets if they're the same
    # For now, we'll use the dataset from model1 for merged embeddings
    merged_embeddings = compute_embeddings(merged_model, datasets[dataset1_name])
    model1_embeddings = compute_embeddings(model1_net, datasets[dataset1_name])
    model2_embeddings = compute_embeddings(model2_net, datasets[dataset2_name])
    
    logger.info("Embedding shapes: merged=%s, model1=%s, model2=%s",
                merged_embeddings.shape, model1_embeddings.shape, model2_embeddings.shape)
    
    # Setup training
    optimizer = torch.optim.Adam(
        partition_layer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    
    # Metadata
    metadata = {
        "hidden_dim": merged_embeddings.size(1),
        "num_nodes": merged_embeddings.size(0),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "sparsity_weight": sparsity_weight,
        "temperature": temperature,
        "num_epochs": num_epochs,
        "seed": seed,
        "dataset1": dataset1_name,
        "dataset2": dataset2_name,
        "model1_metadata": model1_metadata,
        "model2_metadata": model2_metadata,
    }
    
    # Training logs
    logs = collections.defaultdict(list)
    best_total_loss = float('inf')
    best_val_accs = torch.tensor([0. for _ in models], device=merged_embeddings.device)
    
    logger.info("Starting partition layer training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train step
        total_loss, loss1, loss2, sparse_loss = train_step(
            partition_layer,
            merged_embeddings,
            model1_embeddings,
            model2_embeddings,
            optimizer,
            criterion,
            sparsity_weight
        )
        
        # Evaluate with accuracies
        eval_loss, eval_loss1, eval_loss2, mask_mean, mask_std, train_accs, val_accs, test_accs = evaluate_partition(
            partition_layer,
            merged_model,
            models,
            datasets,
            source_metadata,
            masks,
            criterion
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logs["epoch"].append(epoch)
        logs["train_total_loss"].append(total_loss)
        logs["train_loss1"].append(loss1)
        logs["train_loss2"].append(loss2)
        logs["train_sparsity_loss"].append(sparse_loss)
        logs["eval_total_loss"].append(eval_loss)
        logs["eval_loss1"].append(eval_loss1)
        logs["eval_loss2"].append(eval_loss2)
        logs["mask_mean"].append(mask_mean)
        logs["mask_std"].append(mask_std)
        logs["epoch_time"].append(epoch_time)
        
        # Log accuracies for each model
        for i, train_acc in enumerate(train_accs):
            logs[f"train_acc_{i}"].append(train_acc.item())
        for i, val_acc in enumerate(val_accs):
            logs[f"val_acc_{i}"].append(val_acc.item())
        for i, test_acc in enumerate(test_accs):
            logs[f"test_acc_{i}"].append(test_acc.item())
        
        # Save best checkpoint based on validation accuracies
        # We want both validation accuracies to improve
        if torch.all(val_accs > best_val_accs):
            best_val_accs = val_accs.clone()
            best_total_loss = eval_loss
            logs["best_epoch"] = epoch
            logs["best_total_loss"] = eval_loss
            logs["best_loss1"] = eval_loss1
            logs["best_loss2"] = eval_loss2
            logs["best_val_accs"] = val_accs.tolist()
            logs["best_test_accs"] = test_accs.tolist()
            
            save(save_path, partition_layer=partition_layer, metadata=metadata)
            logger.info(
                "Saved best checkpoint (epoch=%d, loss=%.6f, val_accs=%s, test_accs=%s)",
                epoch, eval_loss, val_accs.tolist(), test_accs.tolist()
            )
        
        # Periodic logging
        if (epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            logger.info(
                "Epoch %d/%d | TrainAcc=%s ValAcc=%s TestAcc=%s | "
                "Loss=%.6f (L1=%.6f, L2=%.6f) | Mask: mean=%.3f, std=%.3f | Time=%.2fs",
                epoch + 1, num_epochs,
                train_accs.tolist(),
                val_accs.tolist(),
                test_accs.tolist(),
                eval_loss, eval_loss1, eval_loss2,
                mask_mean, mask_std,
                epoch_time
            )
    
    total_time = time.time() - start_time
    logs["total_training_time"] = total_time
    
    # Save final logs
    save(save_path, logs=logs)
    
    logger.info("Training complete in %.2fs", total_time)
    logger.info("Best loss: %.6f at epoch %d", best_total_loss, logs["best_epoch"])
    logger.info("Best val accuracies: %s", logs["best_val_accs"])
    logger.info("Best test accuracies: %s", logs["best_test_accs"])
    
    # Print final mask statistics
    with torch.no_grad():
        binary_mask = partition_layer.get_binary_mask()
        logger.info("Final binary mask: %d ones, %d zeros (%.2f%% ones)",
                    int(binary_mask.sum().item()),
                    int((1 - binary_mask).sum().item()),
                    100 * float(binary_mask.mean().item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train partition layer for merged GNN model"
    )
    parser.add_argument(
        "--merged-model-path",
        type=Path,
        required=True,
        help="Path to merged model checkpoint"
    )
    parser.add_argument(
        "--model1-path",
        type=Path,
        required=True,
        help="Path to first individual model checkpoint"
    )
    parser.add_argument(
        "--model2-path",
        type=Path,
        required=True,
        help="Path to second individual model checkpoint"
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        required=True,
        help="Path to save partition layer"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WD,
        help="Weight decay"
    )
    parser.add_argument(
        "--sparsity-weight",
        type=float,
        default=0.0,
        help="Weight for sparsity regularization (encourages binary masks)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMP,
        help="Temperature for sigmoid (lower = more binary)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    utils.__init__randomness__(args.seed)
    
    device = utils.get_device()
    logger.info("Using device: %s", device)
    
    # Load models with metadata
    logger.info("Loading model 1 from %s", args.model1_path)
    model1 = utils.load_models(
        args.model1_path,
        task=utils.Task.NodeClassification,
        device=device
    )
    
    logger.info("Loading model 2 from %s", args.model2_path)
    model2 = utils.load_models(
        args.model2_path,
        task=utils.Task.NodeClassification,
        device=device
    )
    
    logger.info("Loading merged model from %s", args.merged_model_path)
    merged_model, merged_metadata = utils.load_models(
        args.merged_model_path,
        task=utils.Task.NodeClassification,
        device=device
    )
    
    # Extract metadata
    model1_net, model1_metadata = model1
    model2_net, model2_metadata = model2
    
    # Verify models have compatible hidden dimensions
    hidden_dim = merged_metadata.get('hidden_dim')
    assert hidden_dim == model1_metadata.get('hidden_dim'), \
        "Model 1 hidden dim doesn't match merged model"
    assert hidden_dim == model2_metadata.get('hidden_dim'), \
        "Model 2 hidden dim doesn't match merged model"
    
    logger.info("All models have hidden_dim=%d", hidden_dim)
    
    # Load datasets from metadata (following merge script pattern)
    models = [model1, model2]
    dataset_names: set[str] = set([metadata['dataset'] for _, metadata in models])
    logger.info("Found datasets: %s", dataset_names)
    
    # Load all unique datasets
    datasets: dict[str, Any] = {
        ds: load_dataset(Path('artifacts/datasets') / (ds + ".pt")) 
        for ds in dataset_names
    }
    
    # Extract just the dataset objects and masks (remove other metadata)
    # load_dataset returns (ds, num_nodes, num_labels, input_dim)
    dataset_objs = {ds: datasets[ds][0].to(device) for ds in datasets}
    
    # Create masks using the same logic as merge script
    # We need to get masks from one of the datasets (assuming they're the same)
    masks, _ = make_label_masks(*(datasets[
        list(dataset_names)[0]
    ][:-1]))
    
    # Move masks to device
    masks = tuple(m.to(device) for m in masks)
    
    logger.info("Loaded %d dataset(s) and created masks", len(dataset_objs))
    
    # Create partition layer
    partition_layer = PartitionLayer(
        hidden_dim=hidden_dim,
        temperature=args.temperature
    ).to(device)
    
    logger.info("Created partition layer with %d parameters",
                sum(p.numel() for p in partition_layer.parameters()))
    
    # Train partition layer
    train_partition_layer(
        partition_layer=partition_layer,
        merged_model=merged_model,
        model1=model1,
        model2=model2,
        datasets=dataset_objs,
        masks=masks,
        save_path=args.save_path,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        sparsity_weight=args.sparsity_weight,
        temperature=args.temperature,
        seed=args.seed
    )
    
    logger.info("Training complete!")
