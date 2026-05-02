#!/usr/bin/env python3
"""
Train link-prediction models with the same checkpoint layout as
train_node_classification.py.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import wandb

import task_evaluation
import utils as utils

graph_size = utils.graph_size
save_artifacts = utils.save

SEED = 42
HIDDEN_DIM = 128
LR = 5e-3
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 500
VAL_RATIO = 0.05
TEST_RATIO = 0.1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("link-prediction-train")


def train_model(
    model: torch.nn.Module,
    hidden_dim: int,
    num_layers: int,
    train_data,
    val_data,
    test_data,
    dataset_name: str,
    input_dim: int,
    optimizer,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    model_type: str,
    ckpt_path: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    wandb_project: str,
    wandb_name: str,
):
    used_num_nodes, used_num_edges = graph_size(train_data)
    config = {
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "task": utils.Task.LinkPrediction,
        "dataset": dataset_name,
        "input_dim": input_dim,
        "num_labels": None,
        "train_edges": int((train_data.edge_label == 1).sum().item()),
        "val_edges": int((val_data.edge_label == 1).sum().item()),
        "test_edges": int((test_data.edge_label == 1).sum().item()),
        "used_num_nodes": used_num_nodes,
        "used_num_edges": used_num_edges,
        "num_classes": 1,
        "chosen_class": 0,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "seed": seed,
        "split_mode": "link_prediction",
        "link_split_seed": seed,
        "link_val_ratio": val_ratio,
        "link_test_ratio": test_ratio,
    }

    wandb.init(project=wandb_project, name=wandb_name, config=config)

    logs: dict[str, str | float | list[float] | Optional[int]] = {
        "epochs": [],
        "train_auc": [],
        "val_auc": [],
        "test_auc": [],
        "train_ap": [],
        "val_ap": [],
        "test_ap": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "time_per_epoch": [],
        "best_epoch": None,
        "best_val_auc": 0.0,
        "best_test_auc": 0.0,
        "best_val_ap": 0.0,
        "best_test_ap": 0.0,
        "total_training_time": 0.0,
    }

    best_val_auc = 0.0
    total_time = 0.0
    logger.info("Starting link-prediction training for %s (%s)", dataset_name, model_type)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_metrics = task_evaluation.train_link_prediction_step(model, train_data, optimizer)
        val_metrics = task_evaluation.evaluate_link_prediction(model, val_data)
        test_metrics = task_evaluation.evaluate_link_prediction(model, test_data)
        train_loss, train_auc, train_ap = train_metrics.loss, train_metrics.auc, train_metrics.ap
        val_loss, val_auc, val_ap = val_metrics.loss, val_metrics.auc, val_metrics.ap
        test_loss, test_auc, test_ap = test_metrics.loss, test_metrics.auc, test_metrics.ap

        epoch_time = time.time() - epoch_start
        total_time += epoch_time

        logs["epochs"].append(epoch)
        logs["train_auc"].append(train_auc)
        logs["val_auc"].append(val_auc)
        logs["test_auc"].append(test_auc)
        logs["train_ap"].append(train_ap)
        logs["val_ap"].append(val_ap)
        logs["test_ap"].append(test_ap)
        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_loss)
        logs["test_loss"].append(test_loss)
        logs["time_per_epoch"].append(epoch_time)

        wandb.log({
            "train_auc": train_auc,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "train_ap": train_ap,
            "val_ap": val_ap,
            "test_ap": test_ap,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "epoch": epoch,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            logs["best_epoch"] = epoch
            logs["best_val_auc"] = val_auc
            logs["best_test_auc"] = test_auc
            logs["best_val_ap"] = val_ap
            logs["best_test_ap"] = test_ap

            config["epoch"] = epoch
            config["val_auc"] = val_auc
            config["test_auc"] = test_auc
            config["val_ap"] = val_ap
            config["test_ap"] = test_ap
            save_artifacts(ckpt_path, merged_model=model, metadata=config, logs=None)
            logger.info("Saved best checkpoint to %s (val_auc=%.4f, test_auc=%.4f)", ckpt_path, val_auc, test_auc)

        if (epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            logger.info(
                "Epoch %d/%d | TrainAUC=%.4f ValAUC=%.4f TestAUC=%.4f | TrainAP=%.4f ValAP=%.4f TestAP=%.4f | TrainLoss=%.4f ValLoss=%.4f TestLoss=%.4f | EpochTime=%.2fs",
                epoch,
                num_epochs - 1,
                train_auc,
                val_auc,
                test_auc,
                train_ap,
                val_ap,
                test_ap,
                train_loss,
                val_loss,
                test_loss,
                epoch_time,
            )

    logs["total_training_time"] = total_time
    save_artifacts(ckpt_path, logs=logs)
    wandb.finish()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train a GNN backbone for link prediction.")
    p.add_argument("--dataset", required=True, help="Name of the dataset")
    p.add_argument("--model", required=True, choices=["gcn", "sage", "gat", "gin"], help="Backbone architecture")
    p.add_argument("--data-path", required=True, type=Path, help="Path to dataset file (torch .pt)")
    p.add_argument("--save-path", required=True, type=Path, help="Checkpoint directory")
    p.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=LR, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="Hidden dimension")
    p.add_argument("--num-layers", type=int, default=2, help="Number of GNN conv layers")
    p.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Validation edge ratio for RandomLinkSplit")
    p.add_argument("--test-ratio", type=float, default=TEST_RATIO, help="Test edge ratio for RandomLinkSplit")
    p.add_argument("--wandb-project", type=str, default="gnnmerge-repro", help="Name given to wandb project")
    p.add_argument("--wandb-name", required=True, type=str, help="Name given to wandb run")
    p.add_argument("--seed", type=int, default=SEED, help="Seed used for reproduction")
    args = p.parse_args()

    utils.__init__randomness__(args.seed)
    device = utils.get_device()
    logger.info("Using device: %s", device)

    dataset, num_nodes, _, input_dim = utils.load_dataset(args.data_path)
    train_data, val_data, test_data = task_evaluation.make_link_split(
        dataset,
        args.seed,
        args.val_ratio,
        args.test_ratio,
    )
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    logger.info(
        "Link split: nodes=%d train_pos_edges=%d val_pos_edges=%d test_pos_edges=%d",
        num_nodes,
        int((train_data.edge_label == 1).sum().item()),
        int((val_data.edge_label == 1).sum().item()),
        int((test_data.edge_label == 1).sum().item()),
    )

    model = utils.build_model(args.model, input_dim, None, device, args.hidden_dim, num_layers=args.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_model(
        model=model,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        dataset_name=args.dataset,
        input_dim=input_dim,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        ckpt_path=args.save_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )

    logger.info("training complete")
