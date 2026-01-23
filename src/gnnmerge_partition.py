#!/usr/bin/env python3
"""
Train a partition layer to decompose merged model embeddings.

The partition layer learns a soft binary mask that satisfies:
    Embed(M(x)) * l ≈ Embed(M_1(x))
    Embed(M(x)) * (1 - l) ≈ Embed(M_2(x))

Usage:
    python train_partition_layer.py \
        --merged-model-path /path/to/merged/model \
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

# -------------------------
# Configurable hyperparams
# -------------------------
DEFAULT_SEED = 42
DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-4
DEFAULT_EPOCHS = 10
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
        mask = torch.sigmoid(self.logits / self.temperature)
        return mask, 1 - mask

    def get_binary_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Get hard binary mask for evaluation."""
        with torch.no_grad():
            mask = torch.sigmoid(self.logits / self.temperature)
            return (mask > threshold).float(), (mask <= threshold).float()

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


def compute_embeddings(model: nn.Module, data) -> torch.Tensor:
    """Extract embeddings from model's backbone."""
    with torch.no_grad():
        # if mode has a backbone, use that
        if hasattr(model, 'backbone'):
            embeddings = model.backbone(data)
        else:
            # otherwise, the model is the backbone
            # (common for link prediction tasks for ex.)
            embeddings = model(data)
        return embeddings


def train_step(
    partition_layer: nn.Module,
    merged_embeds: list[torch.Tensor],
    model_embeds: list[torch.Tensor],
    dataset_masks: list[utils.MaskType],
    optimizer,
    criterion,
    sparsity_weight: float = 0.0
) -> tuple[float, float, float, float]:
    partition_layer.train()
    optimizer.zero_grad()

    # soft masks
    masks = partition_layer(merged_embeds)

    # apply masks
    partitioned_embeds = [embed * m.unsqueeze(0) for m, embed in zip(masks, merged_embeds)]

    # reconstruction losses
    losses = [
        criterion(partition_embed[m[0]], model_embed[m[0]])
        for partition_embed, model_embed, m in
        zip(partitioned_embeds, model_embeds, dataset_masks)
    ]
    losses = torch.stack(losses)

    sparsity_loss = torch.tensor([0.], device=utils.get_device())
    if sparsity_weight > 0:
        surprisals = [m * torch.log(m + 1e-8) for m in masks]
        entropy = - sum(surprisals)
        sparsity_loss = entropy.mean()

    total_loss = losses.sum() + sparsity_weight * sparsity_loss
    total_loss.backward()
    optimizer.step()

    return (
        total_loss,
        losses,
        sparsity_loss
    )


def evaluate_partition(
    partition_layer: PartitionLayer,
    merged_embeds: list[torch.Tensor],
    model_embeds: list[torch.Tensor],
    datasets: dict[str, Any],
    dataset_masks: list[utils.MaskType],
    criterion,
    model_metadata
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate partition layer with both reconstruction loss and classification accuracy.
    """
    partition_layer.eval()
    with torch.no_grad():
        # Compute reconstruction losses
        masks = partition_layer.get_binary_mask()
        # masks = partition_layer(merged_embeds)
        partitioned_embeds = [embed * m.unsqueeze(0) for m, embed in zip(masks, merged_embeds)]
        # print([(m == 0).sum(dim=1, dtype=float).mean() for m in merged_embeds])
        # print([(m == 0).sum(dim=1, dtype=float).mean() for m in model_embeds])
        # print([(m == 0).sum(dim=1, dtype=float).mean() for m in partitioned_embeds])
        # print([m.tolist() for m in masks])

        partitioned_diff = [
            torch.isclose(partition_embed[m[1]], model_embed[m[1]])
            for partition_embed, model_embed, m in
            zip(partitioned_embeds, model_embeds, dataset_masks)
        ]
        merged_diff = [
            torch.isclose(merged_embed[m[1]], model_embed[m[1]])
            for merged_embed, model_embed, m in
            zip(merged_embeds, model_embeds, dataset_masks)
        ]

        # print([m.sum(dim=1, dtype=float).mean() for m in merged_diff])
        # print([p.sum(dim=1, dtype=float).mean() for p in partitioned_diff])

        # reconstruction losses
        losses = [
            criterion(partition_embed[m[1]], model_embed[m[1]])
            for partition_embed, model_embed, m in
            zip(partitioned_embeds, model_embeds, dataset_masks)
        ]
        losses = torch.stack(losses)

        mask_mean = masks[0].mean()
        # as the mask is binary, the std dev can be calculated directly as follows
        mask_std = torch.sqrt((mask_mean) * (1 - mask_mean))

        # get merged outputs
        # todo: adapt link classification
        
        predictions = [model.mlp(out).argmax(dim=1) for model, out in zip(models, merged_embeds)]
        predictions = torch.stack(predictions)
        labels = torch.stack(
            [datasets[meta['dataset']].y for meta in model_metadata['source_models']]
        )

        # predictions shape: [2, num_nodes]
        # labels shape: [2, num_nodes]
        # masks shape: [3, 2, num_nodes]

        masks_tensor = torch.cat([torch.stack(m).unsqueeze(1) for m in dataset_masks], dim=1).to(utils.get_device())

        all_counts = [torch.unique(l, return_counts=True) for l in labels.unsqueeze(0) * masks_tensor]
        logging.info(f"labels distribution: {[dict(zip(unique.tolist(), counts.tolist())) for (unique, counts) in all_counts]}")
        all_counts = [torch.unique(p, return_counts=True) for p in predictions.unsqueeze(0) * masks_tensor]
        logging.info(f"preds distribution: {[dict(zip(unique.tolist(), counts.tolist())) for (unique, counts) in all_counts]}")

        
        # shape: [1, num_models, num_nodes]
        correct = (predictions == labels).float().unsqueeze(0)

        # apply mask and reduce across (train, val, test) on all models
        print(masks_tensor.shape)
        print((correct * masks_tensor).shape)
        masked_correct = (correct * masks_tensor).sum(dim=2)  # [3, num_models]
        mask_sizes = masks_tensor.sum(dim=2)
        accuracies = masked_correct / mask_sizes.clamp(min=1)

        # split into train, val, test
        train_accs = accuracies[0]
        val_accs = accuracies[1]
        test_accs = accuracies[2]

        return (
            losses.sum(),
            losses,
            mask_mean,
            mask_std,
            train_accs,
            val_accs,
            test_accs
        )


def train_partition_layer(
    partition_layer: PartitionLayer,
    merged_model: nn.Module,
    models: list[nn.Module],
    datasets: dict[str, Any],
    masks: tuple,
    save_path: Path,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    sparsity_weight: float,
    temperature: float,
    seed: int,
    merged_model_metadata: dict[str, Any]
):
    """Train the partition layer to decompose merged model embeddings."""

    models = [model.eval() for model in models]
    merged_model = merged_model.eval()

    # TODO: adapt for diff datasets
    merged_embeds = [compute_embeddings(merged_model, datasets[metadata['dataset']])
                         for metadata in merged_model_metadata['source_models']]
    model_embeds = [compute_embeddings(model, datasets[metadata['dataset']])
                    for model, metadata in zip(models, merged_model_metadata['source_models'])]

    logger.info("Embedding shapes: merged=%s, model=%s",
                [e.shape for e in merged_embeds], [e.shape for e in model_embeds])

    # Setup training
    optimizer = torch.optim.Adam(
        partition_layer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    # Metadata
    metadata = {
        "num_epochs": num_epochs,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "sparsity_weight": sparsity_weight,
        "temperature": temperature,
        "merged_metadata": merged_model_metadata
    }

    # Training logs
    logs = collections.defaultdict(list)
    best_total_loss = float('inf')
    best_val_accs = torch.tensor([0. for _ in models], device=utils.get_device())

    logger.info("Starting partition layer training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train step
        total_loss, train_losses, sparse_loss = train_step(
            partition_layer,
            merged_embeds,
            model_embeds,
            masks,
            optimizer,
            criterion,
            sparsity_weight
        )

        # TODO: adapt for link prediction
        val_loss, val_losses, mask_mean, mask_std, train_accs, val_accs, test_accs = evaluate_partition(
            partition_layer,
            merged_embeds,
            model_embeds,
            datasets,
            masks,
            criterion,
            merged_model_metadata
        )

        epoch_time = time.time() - epoch_start

        # Log metrics
        logs["epoch"].append(epoch)
        logs["train_total_loss"].append(total_loss.item())
        logs["train_sparsity_loss"].append(sparse_loss.item())
        logs["eval_total_loss"].append(val_loss.item())
        logs["mask_mean"].append(mask_mean.item())
        logs["mask_std"].append(mask_std.item())
        logs["epoch_time"].append(epoch_time)

        for i, tl in enumerate(train_losses):
            logs[f"train_loss_{i}"].append(tl.item())

        for i, vl in enumerate(val_losses):
            logs[f"val_loss_{i}"].append(vl.item())

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
            best_total_loss = val_loss
            logs["best_epoch"] = epoch
            logs["best_total_loss"] = val_loss.item()
            logs["best_train_losses"] = train_losses.tolist()
            logs["best_val_losses"] = val_losses.tolist()
            logs["best_val_accs"] = val_accs.tolist()
            logs["best_test_accs"] = test_accs.tolist()

            save(save_path, partition_layer=partition_layer, metadata=metadata)
            logger.info(
                "Saved best checkpoint (epoch=%d, loss=%.6f, val_accs=%s, test_accs=%s)",
                epoch, val_loss, val_accs.tolist(), test_accs.tolist()
            )
        
        logger.info(
            "Epoch %d/%d | TrainAcc=%s ValAcc=%s TestAcc=%s | "
            "Loss=%.6f | Mask: mean=%.3f, std=%.3f | Time=%.2fs",
            epoch + 1, num_epochs,
            train_accs.tolist(),
            val_accs.tolist(),
            test_accs.tolist(),
            val_loss, 
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
        partition_layer.load_state_dict(torch.load(save_path / 'partition_layer.pt'))
        binary_mask = partition_layer.get_binary_mask()[0]
        logger.info("Final binary mask: %s", binary_mask)
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

    logger.info("Loading merged model from %s", args.merged_model_path)
    merged_model, merged_metadata = utils.load_models(
        args.merged_model_path,
        task=utils.Task.NodeClassification,
        device=device
    )
    hidden_dim = merged_metadata.get('hidden_dim')


    source_model_metadata = merged_metadata['source_models']
    models = []
    for idx, metadata in enumerate(source_model_metadata):
        models.append(
            utils.load_model_raw(
                args.merged_model_path / f"model_{idx}.pt",
                metadata,
                device=device
            ))

        assert hidden_dim == metadata.get('hidden_dim'), \
            f"Model {idx} has different hidden dim from merged model ({metadata.get('hidden_dim')} vs {hidden_dim})"

    logger.info("All models have hidden_dim=%d", hidden_dim)

    # Load datasets from metadata (following merge script pattern)
    dataset_names: list[str] = [metadata['dataset'] for metadata in source_model_metadata]
    # Load all unique datasets
    datasets: dict[str, Any] = {
        ds: utils.load_dataset(Path('artifacts/datasets') / (ds + ".pt"))
        for ds in set(dataset_names)
    }
    dataset_objs = {ds: datasets[ds][0].to(device) for ds in set(dataset_names)}

    # create masks for all datasets
    all_masks = {}
    model_masks = []
    for metadata in source_model_metadata:
        dataset_name = metadata['dataset']
        # no point in doing repeated computations
        if dataset_name not in all_masks:
            masks = utils.make_label_masks(*(datasets[dataset_name][:-1]), num_classes=metadata['num_classes'])
            all_masks[dataset_name] = masks

        # pick mask based on class chosen
        model_masks.append(all_masks[dataset_name][metadata['chosen_class']])

    # Move masks to device
    model_masks = [tuple(m.to(device) for m in mask) for mask in masks]

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
        models=models,
        datasets=dataset_objs,
        masks=model_masks,
        save_path=args.save_path,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        sparsity_weight=args.sparsity_weight,
        temperature=args.temperature,
        seed=args.seed,
        merged_model_metadata=merged_metadata
    )

    logger.info("Training complete!")
    
