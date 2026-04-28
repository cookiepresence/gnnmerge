from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling


@dataclass(frozen=True)
class LinkPredictionResult:
    auc: float
    ap: float
    loss: float


@dataclass(frozen=True)
class SplitMetrics:
    train: float
    val: float
    test: float

def node_classification_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if mask.sum().item() == 0:
        return 0.0
    predictions = logits.argmax(dim=1)
    correct = int((predictions[mask] == labels[mask]).sum().item())
    total = int(mask.sum().item())
    return float(correct) / float(total)


@torch.no_grad()
def evaluate_node_classification(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    data,
    masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> SplitMetrics:
    backbone.eval()
    head.eval()
    output = backbone(data)
    logits = head(output)
    train_mask, val_mask, test_mask = masks

    return SplitMetrics(
        train=node_classification_accuracy(logits, data.y, train_mask),
        val=node_classification_accuracy(logits, data.y, val_mask),
        test=node_classification_accuracy(logits, data.y, test_mask),
    )


def make_link_split(data, seed: int, val_ratio: float, test_ratio: float):
    # RandomLinkSplit uses torch's global RNG internally.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    transform = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=False,
        num_val=val_ratio,
        num_test=test_ratio,
    )
    return transform(data)


def link_prediction_logits(
    model: torch.nn.Module,
    data,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = model(data)
    pos_logits = model.decode(z, pos_edge_index)
    neg_logits = model.decode(z, neg_edge_index)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([
        torch.ones(pos_logits.size(0), device=logits.device),
        torch.zeros(neg_logits.size(0), device=logits.device),
    ])
    return logits, labels


def link_prediction_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    probs = logits.detach().sigmoid().cpu()
    labels_cpu = labels.detach().cpu()
    return (
        float(roc_auc_score(labels_cpu, probs)),
        float(average_precision_score(labels_cpu, probs)),
    )


def train_link_prediction_step(model: torch.nn.Module, data, optimizer) -> LinkPredictionResult:
    model.train()
    optimizer.zero_grad()

    pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1),
    )
    logits, labels = link_prediction_logits(model, data, pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()

    auc, ap = link_prediction_metrics(logits, labels)
    return LinkPredictionResult(auc=auc, ap=ap, loss=float(loss.item()))


@torch.no_grad()
def evaluate_link_prediction(model: torch.nn.Module, data) -> LinkPredictionResult:
    model.eval()
    pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
    neg_edge_index = data.edge_label_index[:, data.edge_label == 0]
    if neg_edge_index.numel() == 0:
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
        )

    logits, labels = link_prediction_logits(model, data, pos_edge_index, neg_edge_index)
    loss = float(F.binary_cross_entropy_with_logits(logits, labels).item())
    auc, ap = link_prediction_metrics(logits, labels)
    return LinkPredictionResult(auc=auc, ap=ap, loss=loss)


@torch.no_grad()
def evaluate_link_prediction_splits(
    model: torch.nn.Module,
    train_data,
    val_data,
    test_data,
) -> SplitMetrics:
    train_metrics = evaluate_link_prediction(model, train_data)
    val_metrics = evaluate_link_prediction(model, val_data)
    test_metrics = evaluate_link_prediction(model, test_data)
    return SplitMetrics(
        train=train_metrics.auc,
        val=val_metrics.auc,
        test=test_metrics.auc,
    )
