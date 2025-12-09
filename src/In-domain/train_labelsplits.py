#!/usr/bin/env python3
"""
Train label-split models (cleaned & modernized).

Usage (example):
    python train_label_split.py \
        --dataset mydataset \
        --model gcn \
        --data-path /path/to/dataset.pt \
        --model1-save /path/to/model1.pt \
        --model2-save /path/to/model2.pt \
        --logs-dir /path/to/logs
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import random

import torch
import torch.nn as nn
import numpy as np
import wandb

from models import GCNBackbone, GNNComplete, GNNMLP, SageBackbone

# fix for https://github.com/snap-stanford/ogb/issues/497
###
import torch
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])
####

# -------------------------
# Configurable hyperparams
# -------------------------
SEED = 42
HIDDEN_DIM = 128
LR = 5e-3
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 500
CHECKPOINT_EXT = ".pt"
LOG_TIME_FORMAT = "%Y%m%d_%H%M%S"

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("label-split-train")

def __init__randomness(seed):
    import os
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(path: Path):
    """
    Load dataset with torch.load (map to CPU) and return dataset and metadata.
    Note: dataset is returned on CPU. Caller may move it to `device`.
    """
    ds = torch.load(str(path), map_location="cpu")
    num_nodes = int(ds.num_nodes)
    num_labels = int(len(ds.label_names))
    input_dim = int(ds.x.size(1))
    logger.info("Loaded dataset from %s", path)
    logger.info("Nodes=%d; Labels=%d; InputDim=%d", num_nodes, num_labels, input_dim)
    return ds, num_nodes, num_labels, input_dim


def build_models(
        model_name: str, input_dim: int, num_labels: int, device: torch.device,
        hidden_dim: int = HIDDEN_DIM, lr: float = LR, weight_decay: float = WEIGHT_DECAY
) -> tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, torch.optim.Optimizer, nn.Module]:
    """
    Construct two models for the label split, optimizers and criterion.
    """
    if model_name == "gcn":
        b1 = GCNBackbone(input_dim, hidden_dim).to(device)
        b2 = GCNBackbone(input_dim, hidden_dim).to(device)
    elif model_name == "sage":
        b1 = SageBackbone(input_dim, hidden_dim).to(device)
        b2 = SageBackbone(input_dim, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name!r}")

    # split labels between two MLP heads
    head1_out = (num_labels + 1) // 2
    head2_out = num_labels // 2

    m1 = GNNComplete(b1, GNNMLP(hidden_dim, head1_out)).to(device)
    m2 = GNNComplete(b2, GNNMLP(hidden_dim, head2_out)).to(device)

    opt1 = torch.optim.Adam(m1.parameters(), lr=lr, weight_decay=weight_decay)
    opt2 = torch.optim.Adam(m2.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    return m1, m2, opt1, opt2, criterion


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


def train_step(model: torch.nn.Module, data, train_mask: torch.Tensor, optimizer, criterion) -> tuple[float, torch.Tensor]:
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item()), out


def evaluate(model: torch.nn.Module, data, mask: torch.Tensor, criterion) -> tuple[float, float]:
    """
    Returns (accuracy, loss). If mask has zero samples, returns (0.0, 0.0).
    """
    model.eval()
    out = model(data)
    if mask.sum().item() == 0:
        return 0.0, 0.0
    loss = float(criterion(out[mask], data.y[mask]).item())
    pred = out.argmax(dim=1)
    correct = int((pred[mask] == data.y[mask]).sum().item())
    total = int(mask.sum().item())
    acc = float(correct) / float(total) if total > 0 else 0.0
    return acc, loss


def train_model(
    model_id: int,
    model: torch.nn.Module,
    data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    optimizer,
    criterion,
    dataset_name: str,
    model_name: str,
    ckpt_path: Path,
    logs_dir: Path,
    num_epochs: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    wandb_project: str,
    seed: int
) -> dict:
    """
    Train one model and save the best checkpoint based on validation accuracy.
    Returns the logs dictionary.
    """
    timestamp = datetime.now().strftime(LOG_TIME_FORMAT)

    _ = wandb.init(
            project=wandb_project,
            name=f"{dataset_name}_{model_name}_model{model_id}",
            config={
                "model_id": model_id,
                "dataset": dataset_name,
                "model_type": model_name,
                "hidden_dim": hidden_dim,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "train_nodes": train_mask.sum().item(),
                "val_nodes": val_mask.sum().item(),
                "test_nodes": test_mask.sum().item(),
                "seed": seed
            }
        )

    logs: dict[str, str | float | list[float] | Optional[int]] = {
        "model_name": f"{dataset_name}_{model_name}_{model_id}",
        "epochs": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "time_per_epoch": [],
        "best_epoch": None,
        "best_val_acc": 0.0,
        "best_test_acc": 0.0,
        "total_training_time": 0.0,
    }

    best_val_acc = 0.0
    total_time = 0.0

    logger.info("Starting training for model %d (%s)", model_id, logs["model_name"])

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss, _ = train_step(model, data, train_mask, optimizer, criterion)

        train_acc, train_loss = evaluate(model, data, train_mask, criterion)
        val_acc, val_loss = evaluate(model, data, val_mask, criterion)
        test_acc, test_loss = evaluate(model, data, test_mask, criterion)

        epoch_time = time.time() - epoch_start
        total_time += epoch_time

        logs["epochs"].append(epoch)
        logs["train_acc"].append(train_acc)
        logs["val_acc"].append(val_acc)
        logs["test_acc"].append(test_acc)
        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_loss)
        logs["test_loss"].append(test_loss)
        logs["time_per_epoch"].append(epoch_time)

        wandb.log({
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "epoch": epoch,
        })

        # checkpoint on improved validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logs["best_epoch"] = epoch
            logs["best_val_acc"] = val_acc
            logs["best_test_acc"] = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "epoch": epoch,
                },
                str(ckpt_path),
            )
            logger.info("Saved best checkpoint for model %d to %s (val_acc=%.4f, test_acc=%.4f)", model_id, ckpt_path, val_acc, test_acc)

        if (epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            logger.info(
                "Epoch %d/%d | TrainAcc=%.4f ValAcc=%.4f TestAcc=%.4f | TrainLoss=%.4f ValLoss=%.4f TestLoss=%.4f | EpochTime=%.2fs",
                epoch, num_epochs - 1, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, epoch_time,
            )

    logs["total_training_time"] = total_time

    # save logs json
    log_file = logs_dir / f"{dataset_name}_{model_name}_{model_id}_{timestamp}.json"
    log_file.write_text(json.dumps(logs, indent=2))
    logger.info("Saved logs for model %d to %s", model_id, log_file)

    wandb.finish()
    
    return logs


def train_both(
    dataset,
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    masks: MaskType,
    classes: ClassType,
    optimizer1,
    optimizer2,
    criterion,
    dataset_name: str,
    model_name: str,
    ckpt1: Path,
    ckpt2: Path,
    logs_dir: Path,
    num_epochs: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    wandb_project: str,
    seed: int
) -> tuple[dict, dict]:
    (train_mask1, train_mask2, val_mask1, val_mask2, test_mask1, test_mask2) = masks
    classes1, classes2 = classes

    logger.info("Training model 1 (classes: %s)", sorted(list(classes1)))
    logs1 = train_model(
        model_id=1,
        model=model1,
        data=dataset,
        train_mask=train_mask1,
        val_mask=val_mask1,
        test_mask=test_mask1,
        optimizer=optimizer1,
        criterion=criterion,
        dataset_name=dataset_name,
        model_name=model_name,
        ckpt_path=ckpt1,
        logs_dir=logs_dir,
        num_epochs=num_epochs,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
        wandb_project=wandb_project,
        seed=seed
    )

    logger.info("Training model 2 (classes: %s)", sorted(list(classes2)))
    logs2 = train_model(
        model_id=2,
        model=model2,
        data=dataset,
        train_mask=train_mask2,
        val_mask=val_mask2,
        test_mask=test_mask2,
        optimizer=optimizer2,
        criterion=criterion,
        dataset_name=dataset_name,
        model_name=model_name,
        ckpt_path=ckpt2,
        logs_dir=logs_dir,
        num_epochs=num_epochs,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
        wandb_project=wandb_project,
        seed=seed
    )

    # summary
    logger.info("Model 1 best val_acc=%.4f (epoch=%s), test@best=%.4f", logs1["best_val_acc"], logs1["best_epoch"], logs1["best_test_acc"])
    logger.info("Model 2 best val_acc=%.4f (epoch=%s), test@best=%.4f", logs2["best_val_acc"], logs2["best_epoch"], logs2["best_test_acc"])

    return logs1, logs2


def print_split_summary(dataset, masks, classes):
    (train_mask1, train_mask2, val_mask1, val_mask2, test_mask1, test_mask2) = masks
    classes1, classes2 = classes

    logger.info("Label split info: total_classes=%d", len(classes1) + len(classes2))
    logger.info("Model1 classes: %s", sorted(list(classes1)))
    logger.info("Model2 classes: %s", sorted(list(classes2)))

    def unique_labels(tmask):
        return torch.unique(dataset.y[tmask]).tolist()

    logger.info("Model1 - Train labels: %s, Val labels: %s, Test labels: %s, Counts (T/V/Te)=(%d/%d/%d)",
                unique_labels(train_mask1), unique_labels(val_mask1), unique_labels(test_mask1),
                int(train_mask1.sum().item()), int(val_mask1.sum().item()), int(test_mask1.sum().item()))

    logger.info("Model2 - Train labels: %s, Val labels: %s, Test labels: %s, Counts (T/V/Te)=(%d/%d/%d)",
                unique_labels(train_mask2), unique_labels(val_mask2), unique_labels(test_mask2),
                int(train_mask2.sum().item()), int(val_mask2.sum().item()), int(test_mask2.sum().item()))

    total_train = int(train_mask1.sum().item() + train_mask2.sum().item())
    total_val = int(val_mask1.sum().item() + val_mask2.sum().item())
    total_test = int(test_mask1.sum().item() + test_mask2.sum().item())
    logger.info("Totals - Train: %d, Val: %d, Test: %d, Dataset nodes: %d", total_train, total_val, total_test, int(dataset.num_nodes))


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train label-split models (clean version)")
    p.add_argument("--dataset", required=True, help="Name of the dataset (used for logging filenames)")
    p.add_argument("--model", required=True, choices=["gcn", "sage"], help="Backbone architecture")
    p.add_argument("--data-path", required=True, type=Path, help="Path to dataset file (torch .pt)")
    p.add_argument("--model1-save", required=True, type=Path, help="Checkpoint path for model 1")
    p.add_argument("--model2-save", required=True, type=Path, help="Checkpoint path for model 2")
    p.add_argument("--logs-dir", required=True, type=Path, help="Directory to save training logs")
    p.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    p.add_argument("--lr", type=int, default=LR, help="Learning rate for model")
    p.add_argument("--weight-decay", type=int, default=WEIGHT_DECAY, help="Weight decay of model")
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="Number of hidden dims")
    p.add_argument("--wandb-project", type=str, default="gnnmerge-repro", help="Name given to wandb project")
    p.add_argument("--seed", type=int, default=SEED, help="seed used for reproduction")
    args = p.parse_args()

    __init__randomness(args.seed)
    
    device = get_device()
    logger.info("Using device: %s", device)

    dataset, N, labels, input_dim = load_dataset(args.data_path)
    model1, model2, opt1, opt2, criterion = build_models(args.model, input_dim, labels, device, args.hidden_dim, args.lr, args.weight_decay)

    masks, classes = make_label_masks(dataset, N, labels)

    # Move dataset and masks to chosen device for training/inference
    dataset = dataset.to(device)
    masks: MaskType = tuple(m.to(device) for m in masks)

    # print informative split summary
    print_split_summary(dataset, masks, classes)

    # Train both models
    logs1, logs2 = train_both(
        dataset=dataset,
        model1=model1,
        model2=model2,
        masks=masks,
        classes=classes,
        optimizer1=opt1,
        optimizer2=opt2,
        criterion=criterion,
        dataset_name=args.dataset,
        model_name=args.model,
        ckpt1=args.model1_save,
        ckpt2=args.model2_save,
        logs_dir=args.logs_dir,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        wandb_project=args.wandb_project,
        seed=args.seed
    )

    logger.info("All training complete. Logs saved to %s", args.logs_dir)
