#!/usr/bin/env python3
"""
Train node-classification models on datasets that are split by number of labels.
Each model only sees a subset of the labels, instead of seeing the entire dataset.
The number of splits is given by --num-classes, and all labels are split (nearly)
evenly across all classes to prevent data imbalance. If the number of splits is 1,
it is assumed that you want to train on the entire dataset.

Usage (example):
    python train_label_split.py \
        --dataset mydataset \
        --model gcn \
        --data-path /path/to/dataset.pt \
        --save-path /path/to/model.pt \
        --num-classes 2 --chosen-class 0
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Literal
import random

from ogb.nodeproppred import PygNodePropPredDataset

import torch
import torch.nn as nn
import numpy as np
import wandb

from models import GCNBackbone, GNNComplete, GNNMLP, SageBackbone
import utils as utils

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


def save(path: Path, model: Optional[torch.nn.Module]=None, metadata: Optional[dict[str, Any]]=None, logs: Optional[dict]=None):
    path.mkdir(parents=True, exist_ok=True)
    if model is not None:
        torch.save(model.state_dict(), path / "model.pt")
    if metadata is not None:
        metadata_file = path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
    if logs is not None:
        log_file = path / "logs.json"
        log_file.write_text(json.dumps(logs, indent=2))


def train_step(model: torch.nn.Module, dataset, train_mask: torch.Tensor, optimizer, criterion) -> tuple[float, torch.Tensor]:
    model.train()
    optimizer.zero_grad()
    out = model(dataset)
    loss = criterion(out[train_mask], dataset.y[train_mask])
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
    # model related
    model: torch.nn.Module,
    hidden_dim: int,
    criterion,
    # data related
    dataset,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    dataset_name: str,
    input_dim: int,
    num_labels: int,
    num_classes: int,
    chosen_class: int,
    # training related
    optimizer,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    # misc
    model_type: str,
    ckpt_path: Path,
    seed: int,
    wandb_project: str,
    wandb_name: str
):
    """
    Train one model and save the best checkpoint based on validation accuracy.
    Returns the logs dictionary.
    """
    timestamp = datetime.now().strftime(LOG_TIME_FORMAT)

    ckpt_path = ckpt_path
    config = {
        # model related
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "task": utils.Task.NodeClassification, # since the task is node classification
        # dataset related
        "dataset": dataset_name,
        "train_nodes": train_mask.sum().item(),
        "val_nodes": val_mask.sum().item(),
        "test_nodes": test_mask.sum().item(),
        "num_classes": num_classes,
        "chosen_class": chosen_class,
        "input_dim": input_dim,
        "num_labels": num_labels,
        # training related
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "seed": seed
    }

    _ = wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config
        )

    logs: dict[str, str | float | list[float] | Optional[int]] = {
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

    logger.info("Starting training for model %d / %d (%s)", chosen_class, num_classes, config["model_type"])

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss, _ = train_step(model, dataset, train_mask, optimizer, criterion)

        train_acc, train_loss = evaluate(model, dataset, train_mask, criterion)
        val_acc, val_loss = evaluate(model, dataset, val_mask, criterion)
        test_acc, test_loss = evaluate(model, dataset, test_mask, criterion)

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

            config['epoch'] = epoch
            config['val_acc'] = val_acc
            config['test_acc'] = test_acc

            save(ckpt_path, model=model, metadata=config, logs=None)
            logger.info("Saved best checkpoint to %s (val_acc=%.4f, test_acc=%.4f)", ckpt_path, val_acc, test_acc)

        if (epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            logger.info(
                "Epoch %d/%d | TrainAcc=%.4f ValAcc=%.4f TestAcc=%.4f | TrainLoss=%.4f ValLoss=%.4f TestLoss=%.4f | EpochTime=%.2fs",
                epoch, num_epochs - 1, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, epoch_time,
            )

    logs["total_training_time"] = total_time

    # save logs json
    save(ckpt_path, logs=logs)

    wandb.finish()

def print_split_summary(dataset, mask: utils.MaskType):
    train_mask, val_mask, test_mask = mask

    def unique_labels(tmask):
        return torch.unique(dataset.y[tmask]).tolist()

    logger.info("Model - Train labels: %s, Val labels: %s, Test labels: %s, Counts (T/V/Te)=(%d/%d/%d)",
                unique_labels(train_mask), unique_labels(val_mask), unique_labels(test_mask),
                int(train_mask.sum().item()), int(val_mask.sum().item()), int(test_mask.sum().item()))

    total_train = int(train_mask.sum().item())
    total_val = int(val_mask.sum().item())
    total_test = int(test_mask.sum().item())
    logger.info("Totals - Train: %d, Val: %d, Test: %d, Dataset nodes: %d", total_train, total_val, total_test, int(dataset.num_nodes))


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train node-classification models on datasets that are split by number of labels. Each model only sees a subset of the labels, instead of seeing the entire dataset. The number of splits is given by --num-classes, and all labels are split (nearly) evenly across all classes to prevent data imbalance. If the number of splits is 1, it is assumed that you want to train on the entire dataset.")
    p.add_argument("--dataset", required=True, help="Name of the dataset (used for logging filenames)")
    p.add_argument("--model", required=True, choices=["gcn", "sage"], help="Backbone architecture")
    p.add_argument("--data-path", required=True, type=Path, help="Path to dataset file (torch .pt)")
    p.add_argument("--save-path", required=True, type=Path, help="Checkpoint path for model 1")
    p.add_argument("--num-classes", type=int, default=1, help="number of ways the dataset is split")
    p.add_argument("--chosen-class", type=int, default=0, help="class chosen for training. range: [0, num-classes]")
    p.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    p.add_argument("--lr", type=int, default=LR, help="Learning rate for model")
    p.add_argument("--weight-decay", type=int, default=WEIGHT_DECAY, help="Weight decay of model")
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="Number of hidden dims")
    p.add_argument("--wandb-project", type=str, default="gnnmerge-repro", help="Name given to wandb project")
    p.add_argument("--wandb-name", required=True, type=str, help="Name given to wandb run")
    p.add_argument("--seed", type=int, default=SEED, help="seed used for reproduction")
    args = p.parse_args()

    utils.__init__randomness__(args.seed)

    device = utils.get_device()
    logger.info("Using device: %s", device)

    dataset, num_nodes, num_labels, input_dim = utils.load_dataset(args.data_path)
    dataset = dataset.to(utils.get_device())
    masks = utils.make_label_masks(dataset, num_nodes, num_labels, num_classes=args.num_classes)
    train_mask, val_mask, test_mask = masks[args.chosen_class]
    dataset = dataset.to(device)

    print(dataset.y)

    num_masked_labels: int = utils.labels_in_class(args.chosen_class, num_labels, args.num_classes)
    logger.info(f"num of labels used: {num_masked_labels} (out of {num_labels})")
    model = utils.build_model(args.model, input_dim, num_masked_labels, device, args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_model(
        model=model,
        model_type=args.model,
        hidden_dim=args.hidden_dim,

        dataset=dataset,
        dataset_name=args.dataset,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        input_dim=input_dim,
        num_labels=num_masked_labels,
        num_classes=args.num_classes,
        chosen_class=args.chosen_class,

        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,

        seed=args.seed,
        wandb_project=args.wandb_project,
        ckpt_path=args.save_path,
        wandb_name=args.wandb_name
    )

    logger.info("training complete")
