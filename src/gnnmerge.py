import argparse
import collections
import enum
from functools import partial
import math
import logging
from pathlib import Path
from typing import Iterable, Any, Optional

import torch
import wandb
from torch_geometric.utils import negative_sampling

import task_evaluation
import utils

DEFAULT_SEED = 42
DEFAULT_LR = 5e-2
DEFAULT_WD = 0.
DEFAULT_EPOCHS = 1000
DEFAULT_TRAINING_MODE = "layerwise"
DEFAULT_SOFT_LABEL_TEMPERATURE = 1.0
DEFAULT_KL_TEMPERATURE = 1.0


# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gnn-merge")


class SubsampleMode(enum.StrEnum):
    # Keep roughly ratio-per-class from true labels, sampled uniformly within each class.
    GT_CLASS_STRATIFIED_RANDOM = "gt_class_stratified_random"
    # Keep ratio-per-class from true labels, but bias picks toward parent confidence
    # for that same class (soft_labels[node, class]).
    PARENT_SOFT_LABEL_CLASS_STRATIFIED = "parent_soft_label_class_stratified"
    # Ignore class balance and sample globally from all train nodes.
    RANDOM_GLOBAL = "random_global"
    # Keep ratio-per-class from true labels, but bias picks toward high-entropy
    # parent predictions (more uncertain examples).
    PARENT_ENTROPY_CLASS_STRATIFIED = "parent_entropy_class_stratified"


def pad_dataset_features(data, target_dim: int):
    current_dim = data.x.shape[1]
    if current_dim < target_dim:
        padding = torch.zeros(data.x.shape[0], target_dim - current_dim,
                             dtype=data.x.dtype, device=data.x.device)
        data.x = torch.cat([data.x, padding], dim=1)
        logger.info(f"Padded dataset features from {current_dim} to {target_dim}")
    return data


def pad_model_first_layer(model, current_dim: int, target_dim: int, backbone_attr: str = "backbone"):
    if current_dim >= target_dim:
        return model

    backbone = getattr(model, backbone_attr) if hasattr(model, backbone_attr) else model

    first_layer_name = None
    first_layer = None
    for name, module in backbone.named_children():
        first_layer_name = name
        first_layer = module
        break

    if first_layer is None:
        raise ValueError("Could not find first layer in backbone")

    padding_size = target_dim - current_dim

    if hasattr(first_layer, 'lin'):
        old_weight = first_layer.lin.weight.data
        weight_padding = torch.zeros(old_weight.shape[0], padding_size,
                                     dtype=old_weight.dtype, device=old_weight.device)
        first_layer.lin.weight.data = torch.cat([old_weight, weight_padding], dim=1)
        if hasattr(first_layer, 'in_channels'):
            first_layer.in_channels = target_dim
        logger.info(f"Padded model '{first_layer_name}' layer from {current_dim} to {target_dim}")

    elif isinstance(first_layer, torch.nn.Linear):
        old_weight = first_layer.weight.data
        weight_padding = torch.zeros(old_weight.shape[0], padding_size,
                                     dtype=old_weight.dtype, device=old_weight.device)
        first_layer.weight.data = torch.cat([old_weight, weight_padding], dim=1)
        first_layer.in_features = target_dim
        logger.info(f"Padded model Linear layer from {current_dim} to {target_dim}")
    else:
        logger.warning(f"Unknown first layer type: {type(first_layer)}. Attempting to pad 'weight' attribute.")
        if hasattr(first_layer, 'weight'):
            old_weight = first_layer.weight.data
            weight_padding = torch.zeros(old_weight.shape[0], padding_size,
                                        dtype=old_weight.dtype, device=old_weight.device)
            first_layer.weight.data = torch.cat([old_weight, weight_padding], dim=1)

    return model


def metadata_dataset_key(meta: dict[str, Any]) -> str:
    return meta.get("dataset_key", meta["dataset"])


def metadata_task(meta: dict[str, Any]) -> str:
    return str(meta.get("task", utils.Task.NodeClassification))


def is_link_prediction(meta: dict[str, Any]) -> bool:
    return metadata_task(meta) == str(utils.Task.LinkPrediction)


def source_backbone(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "backbone", model)


make_inductive_subgraph = utils.make_inductive_subgraph
graph_size = utils.graph_size
resolve_split_mode = utils.resolve_split_mode
save = utils.save


def hook_fn(module, input, output, ins, outs, layer_name):
    outs[layer_name] = output.detach().clone()
    ins[layer_name] = tuple(inp.detach().clone() for inp in input)

def register_hooks(
    models: Iterable,
    hook_fn,
    backbone_attr: str = "backbone",
) -> tuple[list[list], list[dict], list[dict]]:
    models = list(models)

    all_hooks: list[list] = []
    all_outs: list[dict] = []
    all_ins: list[dict] = []

    for m_idx, model in enumerate(models):
        parent = getattr(model, backbone_attr) if backbone_attr is not None and hasattr(model, backbone_attr) else model

        model_hooks: list = []
        model_outs: dict = {}
        model_ins: dict = {}

        for name, module in parent.named_modules():
            if name == '' or '.' in name:
                continue
            layer_name = f"{name}"
            handle = module.register_forward_hook(
                partial(hook_fn, outs=model_outs, ins=model_ins, layer_name=layer_name)
            )
            model_hooks.append(handle)

        all_hooks.append(model_hooks)
        all_outs.append(model_outs)
        all_ins.append(model_ins)

    return all_hooks, all_outs, all_ins

# def info_nce_loss(
#     merged_model_outputs,
#     model_output,
#     masks,
#     temperature: float = 1.0,
#     normalize: bool = True,
# ):
#     losses = []
#     for merged, orig, mask in zip(merged_model_outputs, model_output, masks):
#         sel = mask[0]
#         z1 = merged[sel]
#         z2 = orig[sel]

#         n = z1.shape[0]
#         if normalize:
#             z1 = torch.nn.functional.normalize(z1, dim=1)
#             z2 = torch.nn.functional.normalize(z2, dim=1)

#         sim = torch.matmul(z1, z2.T) / temperature
#         positives = torch.arange(0, n, device=sim.device)
#         loss = torch.nn.functional.cross_entropy(sim, positives) # + torch.nn.functional.cross_entropy(sim.T, positives)
#         losses.append(loss / 2)

#     return torch.mean(torch.sum(torch.stack(losses), dim=0))

class SigLIPLoss(torch.nn.Module):
    def __init__(
            self,
            normalize: bool = True,
            init_logit_scale: float = 0.0,
            init_logit_bias: float = 0.0,
            max_logit_scale: float = math.log(100.0),
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.max_logit_scale = max_logit_scale
        self.logit_scale = torch.nn.Parameter(torch.tensor(init_logit_scale, dtype=torch.float32))
        self.logit_bias = torch.nn.Parameter(torch.tensor(init_logit_bias, dtype=torch.float32))

    def forward(self, merged_model_outputs, model_output, masks):
        losses = []

        scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()

        for merged, orig, mask in zip(merged_model_outputs, model_output, masks):
            sel = mask[0]

            z1 = merged[sel]
            z2 = orig[sel]

            n = z1.shape[0]
            if n == 0:
                continue

            if self.normalize:
                z1 = torch.nn.functional.normalize(z1, dim=1)
                z2 = torch.nn.functional.normalize(z2, dim=1)

            logits = torch.matmul(z1, z2.T)
            logits = logits * scale + self.logit_bias

            labels = 2 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1
            loss = -torch.nn.functional.logsigmoid(labels * logits).mean()
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=utils.get_device())

        return torch.mean(torch.stack(losses))

def subsample_train_mask_random_global(
        train_mask: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
):
    # Random-global: choose k from all train nodes, independent of class.
    # This can shift the label distribution compared to the original mask.
    if ratio >= 1.0:
        return train_mask

    new_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    idx = train_mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return new_mask

    k = max(1, int(idx.numel() * ratio))
    k = min(k, idx.numel())
    perm = torch.randperm(idx.numel(), generator=generator, device=generator.device)
    chosen = idx[perm[:k]]
    new_mask[chosen] = True
    return new_mask


def _compute_subsample_k(n: int, ratio: float) -> int:
    # Per bucket we keep at least one node (when bucket is non-empty), capped by n.
    return min(max(1, int(n * ratio)), n)


def _sample_from_indices(
        idx: torch.Tensor,
        k: int,
        generator: torch.Generator,
        scores: Optional[torch.Tensor] = None,
        temperature: float = DEFAULT_SOFT_LABEL_TEMPERATURE,
):
    # Shared "sample k from idx without replacement" primitive.
    # - scores=None: uniform sample.
    # - scores provided: weighted sample via softmax(log(score)/temperature).
    #   Lower temperature sharpens toward high-score nodes.
    if k >= idx.numel():
        return idx
    if scores is None:
        perm = torch.randperm(idx.numel(), generator=generator, device=generator.device)
        return idx[perm[:k]]
    probs = torch.softmax(torch.log(scores.clamp_min(1e-12)) / temperature, dim=0)
    chosen_rel = torch.multinomial(probs, num_samples=k, replacement=False, generator=generator)
    return idx[chosen_rel]


def _subsample_train_mask_class_stratified(
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
        score_fn=None,
        temperature: float = DEFAULT_SOFT_LABEL_TEMPERATURE,
):
    # Class-stratified template:
    # 1) iterate over classes present in train_mask
    # 2) sample k ~= ratio * class_count for each class
    # 3) optional score_fn adds within-class weighting
    # This preserves class coverage while allowing importance bias inside each class.
    if ratio >= 1.0:
        return train_mask
    new_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    for c in labels[train_mask].unique():
        idx = ((labels == c) & train_mask).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        k = _compute_subsample_k(idx.numel(), ratio)
        scores = score_fn(int(c.item()), idx) if score_fn is not None else None
        new_mask[_sample_from_indices(idx, k, generator, scores=scores, temperature=temperature)] = True
    return new_mask


def subsample_train_mask_gt_class_stratified_random(
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
):
    # True-label stratified + uniform within each class.
    return _subsample_train_mask_class_stratified(labels, train_mask, ratio, generator)


def subsample_train_mask_parent_soft_label_class_stratified(
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        ratio: float,
        soft_labels: torch.Tensor,
        generator: torch.Generator,
        soft_label_temperature: float = DEFAULT_SOFT_LABEL_TEMPERATURE,
):
    # True-label stratified, but within each class c, nodes with higher
    # parent-predicted p(y=c) are sampled more often.
    if soft_label_temperature <= 0:
        raise ValueError("soft_label_temperature must be > 0")
    if soft_labels.ndim != 2:
        raise ValueError(f"Expected soft_labels with shape [num_nodes, num_classes], got {tuple(soft_labels.shape)}")
    def score_fn(class_idx: int, idx: torch.Tensor) -> torch.Tensor:
        if class_idx < 0 or class_idx >= soft_labels.shape[1]:
            raise ValueError(
                f"Label class index {class_idx} invalid for soft_labels with {soft_labels.shape[1]} columns"
            )
        return soft_labels[idx, class_idx]
    return _subsample_train_mask_class_stratified(
        labels,
        train_mask,
        ratio,
        generator,
        score_fn=score_fn,
        temperature=soft_label_temperature,
    )


def subsample_train_mask_parent_entropy_class_stratified(
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        ratio: float,
        soft_labels: torch.Tensor,
        generator: torch.Generator,
        soft_label_temperature: float = DEFAULT_SOFT_LABEL_TEMPERATURE,
):
    # True-label stratified, but within each class, sample by prediction entropy.
    # Higher entropy => more uncertain parent prediction => higher sampling weight.
    if soft_label_temperature <= 0:
        raise ValueError("soft_label_temperature must be > 0")
    if soft_labels.ndim != 2:
        raise ValueError(f"Expected soft_labels with shape [num_nodes, num_classes], got {tuple(soft_labels.shape)}")
    def score_fn(_class_idx: int, idx: torch.Tensor) -> torch.Tensor:
        probs = soft_labels[idx].clamp_min(1e-12)
        return -(probs * probs.log()).sum(dim=1).clamp_min(1e-12)
    return _subsample_train_mask_class_stratified(
        labels,
        train_mask,
        ratio,
        generator,
        score_fn=score_fn,
        temperature=soft_label_temperature,
    )


def _binary_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-12, 1.0 - 1e-12)
    return -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())


def make_link_prediction_soft_labels(
        model: torch.nn.Module,
        data,
        embeddings: torch.Tensor,
        seed: int,
) -> dict[str, torch.Tensor]:
    pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
    if pos_edge_index.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=embeddings.device)
        return {
            "edge_index": torch.empty((2, 0), dtype=torch.long, device=embeddings.device),
            "hard_labels": empty.float(),
            "soft_labels": empty.float(),
        }

    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
        )
    finally:
        torch.random.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    hard_labels = torch.cat([
        torch.ones(pos_edge_index.size(1), dtype=torch.float32, device=embeddings.device),
        torch.zeros(neg_edge_index.size(1), dtype=torch.float32, device=embeddings.device),
    ])
    soft_labels = model.decode(embeddings, edge_index).sigmoid().detach()
    return {
        "edge_index": edge_index.detach(),
        "hard_labels": hard_labels.detach(),
        "soft_labels": soft_labels,
    }


def _link_prediction_all_edge_indices(edge_index: torch.Tensor) -> torch.Tensor:
    return torch.arange(edge_index.size(1), device=edge_index.device)


def _subsample_link_prediction_edges_class_stratified(
        hard_labels: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
        score_fn=None,
        temperature: float = DEFAULT_SOFT_LABEL_TEMPERATURE,
) -> torch.Tensor:
    all_idx = torch.arange(hard_labels.numel(), device=hard_labels.device)
    if ratio >= 1.0:
        return all_idx

    selected_parts = []
    for label in hard_labels.unique():
        idx = (hard_labels == label).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        scores = score_fn(idx) if score_fn is not None else None
        selected_parts.append(
            _sample_from_indices(
                idx,
                _compute_subsample_k(idx.numel(), ratio),
                generator,
                scores=scores,
                temperature=temperature,
            )
        )
    return torch.cat(selected_parts) if selected_parts else all_idx[:0]


def subsample_link_prediction_edges_random_global(
        edge_index: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
) -> torch.Tensor:
    # LP analogue of random node sampling: sample uniformly from all candidate edges.
    idx = _link_prediction_all_edge_indices(edge_index)
    if ratio >= 1.0:
        return idx
    return _sample_from_indices(idx, _compute_subsample_k(idx.numel(), ratio), generator)


def subsample_link_prediction_edges_gt_class_stratified_random(
        hard_labels: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
) -> torch.Tensor:
    # LP "classes" are positive and negative candidate edges.
    return _subsample_link_prediction_edges_class_stratified(hard_labels, ratio, generator)


def subsample_link_prediction_edges_parent_soft_label_class_stratified(
        hard_labels: torch.Tensor,
        soft_labels: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
        soft_label_temperature: float = DEFAULT_SOFT_LABEL_TEMPERATURE,
) -> torch.Tensor:
    if soft_label_temperature <= 0:
        raise ValueError("soft_label_temperature must be > 0")

    def score_fn(idx: torch.Tensor) -> torch.Tensor:
        # soft_labels are p(edge exists | u, v). For negative candidates,
        # confidence in the hard edge class is 1 - p(edge exists | u, v).
        return torch.where(hard_labels[idx] > 0.5, soft_labels[idx], 1.0 - soft_labels[idx])

    return _subsample_link_prediction_edges_class_stratified(
        hard_labels,
        ratio,
        generator,
        score_fn=score_fn,
        temperature=soft_label_temperature,
    )


def subsample_link_prediction_edges_parent_entropy_class_stratified(
        hard_labels: torch.Tensor,
        soft_labels: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
        soft_label_temperature: float = DEFAULT_SOFT_LABEL_TEMPERATURE,
) -> torch.Tensor:
    if soft_label_temperature <= 0:
        raise ValueError("soft_label_temperature must be > 0")
    entropy = _binary_entropy(soft_labels)
    return _subsample_link_prediction_edges_class_stratified(
        hard_labels,
        ratio,
        generator,
        score_fn=lambda idx: entropy[idx],
        temperature=soft_label_temperature,
    )


def link_prediction_edges_to_node_mask(
        train_mask: torch.Tensor,
        edge_index: torch.Tensor,
        selected_edges: torch.Tensor,
) -> torch.Tensor:
    node_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    if selected_edges.numel() > 0:
        selected_nodes = edge_index[:, selected_edges].reshape(-1).unique()
        node_mask[selected_nodes] = True
        node_mask &= train_mask
    return node_mask

def compute_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """
    Compute per-parameter and total L2 gradient norms for a model.
    Call after .backward() and before .zero_grad().

    Returns a dict with keys "<param_name>" for each parameter that has a
    gradient, plus "total" for the global L2 norm across all parameters.
    """
    norms: dict[str, float] = {}
    total_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            pnorm = param.grad.detach().norm(2).item()
            norms[name] = pnorm
            total_sq += pnorm ** 2
    norms["total"] = total_sq ** 0.5
    return norms

def train(
        merged_model: torch.nn.Module,
        model_outputs: list[dict],
        model_inputs: list[dict],
        masks: list[utils.MaskType],
        optimizer,
        criterion,
        mse_loss_fn=None,
        contrastive_loss_fn=None,
        kl_loss_fn=None,
):
    merged_model.train()
    optimizer.zero_grad()
    losses = []
    mse_losses = []
    contrastive_losses = []
    kl_losses = []

    for layer in model_outputs[0].keys():
        model_output = [outputs[layer] for outputs in model_outputs]
        model_input  = [inputs[layer]  for inputs  in model_inputs]

        merged_model_layer   = getattr(merged_model, layer)
        merged_model_outputs = [merged_model_layer(*inp) for inp in model_input]

        loss = criterion(merged_model_outputs, model_output, masks)
        losses.append(loss)

        if mse_loss_fn is not None:
            with torch.no_grad():
                mse_losses.append(mse_loss_fn(merged_model_outputs, model_output, masks).item())
        if contrastive_loss_fn is not None:
            with torch.no_grad():
                contrastive_losses.append(contrastive_loss_fn(merged_model_outputs, model_output, masks).item())
        if kl_loss_fn is not None:
            with torch.no_grad():
                kl_losses.append(kl_loss_fn(merged_model_outputs, model_output, masks).item())

    # [loss.backward() for loss in losses]
    torch.stack(losses).sum().backward()
    grad_norms = compute_grad_norms(merged_model)
    optimizer.step()
    return losses, grad_norms, mse_losses, contrastive_losses, kl_losses

def train_e2e(
        merged_model: torch.nn.Module,
        teacher_outputs: list[torch.Tensor],
        datasets: dict[str, Any],
        metadata: list[dict[str, Any]],
        masks: list[utils.MaskType],
        optimizer,
        criterion,
        mse_loss_fn=None,
        contrastive_loss_fn=None,
        kl_loss_fn=None,
):
    merged_model.train()
    optimizer.zero_grad()

    merged_outputs = [merged_model(datasets[metadata_dataset_key(meta)]) for meta in metadata]
    loss = criterion(merged_outputs, teacher_outputs, masks)
    loss.backward()
    grad_norms = compute_grad_norms(merged_model)
    optimizer.step()

    losses = [loss]
    mse_losses = []
    contrastive_losses = []
    kl_losses = []
    if mse_loss_fn is not None:
        with torch.no_grad():
            mse_losses.append(mse_loss_fn(merged_outputs, teacher_outputs, masks).item())
    if contrastive_loss_fn is not None:
        with torch.no_grad():
            contrastive_losses.append(contrastive_loss_fn(merged_outputs, teacher_outputs, masks).item())
    if kl_loss_fn is not None:
        with torch.no_grad():
            kl_losses.append(kl_loss_fn(merged_outputs, teacher_outputs, masks).item())

    return losses, grad_norms, mse_losses, contrastive_losses, kl_losses

def evaluate(
        merged_model,
        models,
        metadata,
        datasets,
        masks,
        source_tasks: list[str],
):
    merged_model.eval()

    train_scores = []
    val_scores = []
    test_scores = []
    aux_metrics: dict[str, list[float]] = collections.defaultdict(list)

    for i, (model, meta, mask, task) in enumerate(zip(models, metadata, masks, source_tasks)):
        if task == str(utils.Task.LinkPrediction):
            metrics = task_evaluation.evaluate_link_prediction_splits(
                merged_model,
                datasets[metadata_dataset_key(meta)],
                datasets[meta["link_val_dataset_key"]],
                datasets[meta["link_test_dataset_key"]],
            )
        elif task == str(utils.Task.NodeClassification):
            metrics = task_evaluation.evaluate_node_classification(
                merged_model,
                model.mlp,
                datasets[metadata_dataset_key(meta)],
                mask,
            )
        else:
            raise ValueError(f"Unsupported task for evaluation: {task}")

        train_scores.append(torch.tensor(metrics.train, device=utils.get_device()))
        val_scores.append(torch.tensor(metrics.val, device=utils.get_device()))
        test_scores.append(torch.tensor(metrics.test, device=utils.get_device()))
        for name, value in getattr(metrics, "aux", {}).items():
            aux_metrics[f"{name}_{i}"].append(value)

    return (
        torch.stack(train_scores).to(utils.get_device()),
        torch.stack(val_scores).to(utils.get_device()),
        torch.stack(test_scores).to(utils.get_device()),
        aux_metrics,
    )
def merge_model(
        merged_model: torch.nn.Module,
        save_path: Path,
        models: list[tuple[torch.nn.Module, dict]],
        datasets,
        masks,
        subsample_ratio: float,
        mse_loss_weight: float,
        contrastive_loss_weight: float,
        kl_loss_weight: float,
        kl_temperature: float,
        learning_rate: float,
        weight_decay: float,
        seed: int,
        num_epochs: int = 1000,
        training_mode: str = DEFAULT_TRAINING_MODE,
        subsample_mode: str = SubsampleMode.GT_CLASS_STRATIFIED_RANDOM.value,
        soft_label_temperature: float = DEFAULT_SOFT_LABEL_TEMPERATURE,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
):
    if mse_loss_weight <= 0 and contrastive_loss_weight <= 0 and kl_loss_weight <= 0:
        raise ValueError("At least one of mse_loss_weight, contrastive_loss_weight, or kl_loss_weight must be > 0")
    if kl_loss_weight > 0 and kl_temperature <= 0:
        raise ValueError("kl_temperature must be > 0 when kl_loss_weight > 0")

    source_metadata: list[dict] = [metadata for _, metadata in models]
    source_tasks = [metadata_task(meta) for meta in source_metadata]
    source_is_lp = [task == str(utils.Task.LinkPrediction) for task in source_tasks]
    source_metrics = ["auc" if is_lp else "acc" for is_lp in source_is_lp]
    models: list[torch.nn.Module] = [model.eval() for model, _ in models]
    with torch.no_grad():
        teacher_outputs = [
            source_backbone(model)(datasets[metadata_dataset_key(metadata)]).detach()
            for model, metadata in zip(models, source_metadata)
        ]
        parent_soft_labels = []
        for model_idx, (model, is_lp, output, meta) in enumerate(
            zip(models, source_is_lp, teacher_outputs, source_metadata)
        ):
            if is_lp:
                lp_soft_labels = make_link_prediction_soft_labels(
                    model,
                    datasets[metadata_dataset_key(meta)],
                    output,
                    seed + model_idx,
                )
                parent_soft_labels.append(lp_soft_labels)
            else:
                parent_soft_labels.append(torch.softmax(model.mlp(output), dim=1).detach())
    model_outputs = []
    model_inputs = []
    if training_mode == "layerwise":
        model_hooks, model_outputs, model_inputs = register_hooks(models, hook_fn)
        _ = [source_backbone(model)(datasets[metadata_dataset_key(metadata)]) for model, metadata in zip(models, source_metadata)]
    elif training_mode != "e2e":
        raise ValueError(f"Unknown training mode: {training_mode}")
    siglip = SigLIPLoss().to(utils.get_device()) if contrastive_loss_weight > 0 else None
    optimizer_params = list(merged_model.parameters())
    if siglip is not None:
        optimizer_params.extend(siglip.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate, weight_decay=weight_decay)

    mse_loss = lambda merged_model_outputs, model_output, masks: torch.sum(
        torch.stack([
            torch.nn.MSELoss(reduction='mean')(merged[mask[0]], orig[mask[0]])
            for merged, orig, mask in zip(merged_model_outputs, model_output, masks)
        ])
    )
    def kl_loss(merged_model_outputs, model_output, masks):
        losses = []
        for merged, orig, mask in zip(merged_model_outputs, model_output, masks):
            sel = mask[0]
            if sel.sum().item() == 0:
                continue
            merged_log_probs = torch.nn.functional.log_softmax(merged[sel] / kl_temperature, dim=-1)
            teacher_probs = torch.nn.functional.softmax(orig[sel] / kl_temperature, dim=-1)
            losses.append(
                torch.nn.functional.kl_div(
                    merged_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (kl_temperature ** 2)
            )
        if not losses:
            return torch.tensor(0.0, device=utils.get_device())
        return torch.sum(torch.stack(losses))

    metadata = {
        "weight_decay": weight_decay,
        "lr": learning_rate,
        "epochs": num_epochs,
        "seed": seed,
        "model_type": source_metadata[0]['model_type'],
        "input_dim": source_metadata[0]['input_dim'],
        "hidden_dim": source_metadata[0]['hidden_dim'],
        "num_layers": source_metadata[0].get('num_layers', 2),
        "mse_loss_weight": mse_loss_weight,
        "contrastive_loss_weight": contrastive_loss_weight,
        "kl_loss_weight": kl_loss_weight,
        "kl_temperature": kl_temperature,
        "subsample_ratio": subsample_ratio,
        "subsample_mode": subsample_mode,
        "soft_label_temperature": soft_label_temperature,
        "training_mode": training_mode,
        "source_models": source_metadata
    }

    logs = collections.defaultdict(list)
    aux_state = {"contrastive_head": siglip.state_dict()} if siglip is not None else None
    save(save_path, None, models, metadata, None, aux_state=aux_state)
    best_val_scores = torch.tensor([0. for _ in models], device=utils.get_device())

    subsampled_masks = []
    sampling_stats: list[dict[str, Any]] = []
    for (train_mask, val_mask, test_mask), soft_labels, meta, task, is_lp in zip(
        masks,
        parent_soft_labels,
        source_metadata,
        source_tasks,
        source_is_lp,
    ):
        # ensure that all datasets are sampled with the same seed.
        gen = torch.Generator(device=utils.get_device())
        gen.manual_seed(seed)

        if is_lp:
            lp_edge_index = soft_labels["edge_index"]
            lp_hard_labels = soft_labels["hard_labels"]
            lp_soft_labels = soft_labels["soft_labels"]
            if subsample_mode == SubsampleMode.GT_CLASS_STRATIFIED_RANDOM.value:
                selected_edges = subsample_link_prediction_edges_gt_class_stratified_random(
                    lp_hard_labels,
                    subsample_ratio,
                    gen,
                )
            elif subsample_mode == SubsampleMode.PARENT_SOFT_LABEL_CLASS_STRATIFIED.value:
                selected_edges = subsample_link_prediction_edges_parent_soft_label_class_stratified(
                    lp_hard_labels,
                    lp_soft_labels,
                    subsample_ratio,
                    gen,
                    soft_label_temperature=soft_label_temperature,
                )
            elif subsample_mode == SubsampleMode.RANDOM_GLOBAL.value:
                selected_edges = subsample_link_prediction_edges_random_global(
                    lp_edge_index,
                    subsample_ratio,
                    gen,
                )
            elif subsample_mode == SubsampleMode.PARENT_ENTROPY_CLASS_STRATIFIED.value:
                selected_edges = subsample_link_prediction_edges_parent_entropy_class_stratified(
                    lp_hard_labels,
                    lp_soft_labels,
                    subsample_ratio,
                    gen,
                    soft_label_temperature=soft_label_temperature,
                )
            else:
                raise ValueError(f"Unknown subsample mode: {subsample_mode}")
            sub_train = link_prediction_edges_to_node_mask(train_mask, lp_edge_index, selected_edges)
            subsampled_masks.append((sub_train, val_mask, test_mask))
            sampling_stats.append(
                {
                    "dataset": meta["dataset"],
                    "task": task,
                    "chosen_class": int(meta.get("chosen_class", 0)),
                    "before": int(train_mask.sum().item()),
                    "sampled": int(sub_train.sum().item()),
                    "before_edges": int(lp_edge_index.size(1)),
                    "sampled_edges": int(selected_edges.numel()),
                }
            )
            continue

        labels = datasets[metadata_dataset_key(meta)].y
        # Dispatch exactly one strategy per source model; resulting sub_train keeps
        # only the selected training nodes while val/test masks stay unchanged.
        if subsample_mode == SubsampleMode.GT_CLASS_STRATIFIED_RANDOM.value:
            sub_train = subsample_train_mask_gt_class_stratified_random(
                labels,
                train_mask,
                subsample_ratio,
                gen,
            )
        elif subsample_mode == SubsampleMode.PARENT_SOFT_LABEL_CLASS_STRATIFIED.value:
            sub_train = subsample_train_mask_parent_soft_label_class_stratified(
                datasets[metadata_dataset_key(meta)].y,
                train_mask,
                subsample_ratio,
                soft_labels,
                gen,
                soft_label_temperature=soft_label_temperature,
            )
        elif subsample_mode == SubsampleMode.RANDOM_GLOBAL.value:
            sub_train = subsample_train_mask_random_global(train_mask, subsample_ratio, gen)
        elif subsample_mode == SubsampleMode.PARENT_ENTROPY_CLASS_STRATIFIED.value:
            sub_train = subsample_train_mask_parent_entropy_class_stratified(
                labels,
                train_mask,
                subsample_ratio,
                soft_labels,
                gen,
                soft_label_temperature=soft_label_temperature,
            )
        else:
            raise ValueError(f"Unknown subsample mode: {subsample_mode}")
        subsampled_masks.append((sub_train, val_mask, test_mask))
        sampling_stats.append(
            {
                "dataset": meta["dataset"],
                "task": task,
                "chosen_class": int(meta.get("chosen_class", 0)),
                "before": int(train_mask.sum().item()),
                "sampled": int(sub_train.sum().item()),
            }
        )
    for stat in sampling_stats:
        if "sampled_edges" in stat:
            logger.info(
                "Sampled train edges/nodes (pre-train) | dataset=%s task=%s class=%d "
                "edges=%d/%d nodes=%d/%d",
                stat["dataset"],
                stat["task"],
                stat["chosen_class"],
                stat["sampled_edges"],
                stat["before_edges"],
                stat["sampled"],
                stat["before"],
            )
        else:
            logger.info(
                "Sampled train nodes (pre-train) | dataset=%s task=%s class=%d sampled=%d/%d",
                stat["dataset"],
                stat["task"],
                stat["chosen_class"],
                stat["sampled"],
                stat["before"],
            )

    # ------------------------------------------------------------------
    # wandb: build a stable per-model metric prefix from dataset metadata
    # e.g.  "cora/2/7"  →  dataset=cora, chosen_class=2, num_classes=7
    # ------------------------------------------------------------------
    use_wandb = wandb_project is not None
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=metadata,
        )
        # Pre-compute prefix strings so we don't recompute every epoch
        # layers (keys) are only known after the first forward pass, handled inside the loop
        model_prefixes = [
            f"{meta['dataset']}/{task}/{meta.get('chosen_class')}/{meta.get('num_classes')}"
            for task, meta in zip(source_tasks, source_metadata)
        ]
        layer_names = list(model_outputs[0].keys()) if training_mode == "layerwise" else ["final"]
        logger.info("wandb run initialised — prefixes: %s", model_prefixes)

    def criterion(merged_model_outputs, model_output, masks):
        loss_terms = []
        if mse_loss_weight > 0:
            loss_terms.append(mse_loss_weight * mse_loss(merged_model_outputs, model_output, masks))
        if contrastive_loss_weight > 0:
            loss_terms.append(contrastive_loss_weight * siglip(merged_model_outputs, model_output, masks))
        if kl_loss_weight > 0:
            loss_terms.append(kl_loss_weight * kl_loss(merged_model_outputs, model_output, masks))
        return torch.stack(loss_terms).sum()

    logger.info("starting model merging...")
    for epoch in range(num_epochs):
        if training_mode == "layerwise":
            train_losses, grad_norms, mse_losses, contrastive_losses, kl_losses = train(
                merged_model, model_outputs, model_inputs, subsampled_masks, optimizer, criterion,
                contrastive_loss_fn=siglip if contrastive_loss_weight > 0 else None,
                mse_loss_fn=mse_loss if mse_loss_weight > 0 else None,
                kl_loss_fn=kl_loss if kl_loss_weight > 0 else None)
        else:
            train_losses, grad_norms, mse_losses, contrastive_losses, kl_losses = train_e2e(
                merged_model, teacher_outputs, datasets, source_metadata, subsampled_masks, optimizer, criterion,
                contrastive_loss_fn=siglip if contrastive_loss_weight > 0 else None,
                mse_loss_fn=mse_loss if mse_loss_weight > 0 else None,
                kl_loss_fn=kl_loss if kl_loss_weight > 0 else None)
        train_scores, val_scores, test_scores, aux_metrics = evaluate(
            merged_model,
            models,
            source_metadata,
            datasets,
            masks,
            source_tasks,
        )

        # ---- internal logs (unchanged) ----
        for i, loss in enumerate(train_losses):
            logs[f"train_loss_{i}"].append(loss.item())
        if mse_loss_weight > 0:
            for i, loss in enumerate(mse_losses):
                logs[f"train_mse_loss_{i}"].append(loss)
        if contrastive_loss_weight > 0:
            for i, loss in enumerate(contrastive_losses):
                logs[f"train_contrastive_loss_{i}"].append(loss)
            logs["contrastive_logit_scale"].append(siglip.logit_scale.detach().item())
            logs["contrastive_logit_bias"].append(siglip.logit_bias.detach().item())
        if kl_loss_weight > 0:
            for i, loss in enumerate(kl_losses):
                logs[f"train_kl_loss_{i}"].append(loss)
        for i, train_score in enumerate(train_scores):
            logs[f"train_{source_metrics[i]}_{i}"].append(train_score.item())
        for i, val_score in enumerate(val_scores):
            logs[f"val_{source_metrics[i]}_{i}"].append(val_score.item())
        for i, test_score in enumerate(test_scores):
            logs[f"test_{source_metrics[i]}_{i}"].append(test_score.item())
        logs["grad_norm_total"].append(grad_norms["total"])

        # ---- wandb logging ----
        if use_wandb:
            wb_log: dict[str, float] = {}

            # Per-model task metrics, namespaced by dataset/task/split/total.
            for i, prefix in enumerate(model_prefixes):
                metric = source_metrics[i]
                wb_log[f"{prefix}/train_{metric}"] = train_scores[i].item()
                wb_log[f"{prefix}/val_{metric}"] = val_scores[i].item()
                wb_log[f"{prefix}/test_{metric}"] = test_scores[i].item()

            # Per-layer losses, also namespaced by dataset/split/total for each model
            for layer_idx, (layer_name, loss) in enumerate(zip(layer_names, train_losses)):
                for i, prefix in enumerate(model_prefixes):
                    wb_log[f"{prefix}/train_loss/{layer_name}"] = loss.item()
            if mse_loss_weight > 0:
                for layer_idx, (layer_name, loss) in enumerate(zip(layer_names, mse_losses)):
                    for i, prefix in enumerate(model_prefixes):
                        wb_log[f"{prefix}/train_mse_loss/{layer_name}"] = loss
            if contrastive_loss_weight > 0:
                for layer_idx, (layer_name, loss) in enumerate(zip(layer_names, contrastive_losses)):
                    for i, prefix in enumerate(model_prefixes):
                        wb_log[f"{prefix}/train_contrastive_loss/{layer_name}"] = loss
                wb_log["contrastive/logit_scale"] = siglip.logit_scale.detach().item()
                wb_log["contrastive/logit_bias"] = siglip.logit_bias.detach().item()
            if kl_loss_weight > 0:
                for layer_idx, (layer_name, loss) in enumerate(zip(layer_names, kl_losses)):
                    for i, prefix in enumerate(model_prefixes):
                        wb_log[f"{prefix}/train_kl_loss/{layer_name}"] = loss

            # Grad norms — logged under grad_norm/<param_name> and grad_norm/total
            # These belong to the merged model, not to any individual source model,
            # so we use a flat namespace rather than the per-model prefix.
            for param_name, norm_val in grad_norms.items():
                # Replace dots in param names (e.g. "backbone.conv1.weight") with
                # slashes so wandb groups them into a clean nested panel.
                key = param_name.replace(".", "/")
                wb_log[f"grad_norm/{key}"] = norm_val

            wandb.log(wb_log, step=epoch)

        # ---- checkpoint ----
        if torch.all(val_scores > best_val_scores):
            best_val_scores = val_scores.clone()
            aux_state = {"contrastive_head": siglip.state_dict()} if siglip is not None else None
            save(save_path, merged_model, None, None, logs, aux_state=aux_state)
            logger.info(
                "Saved best checkpoint to %s (train_scores=%s, val_scores=%s, test_scores=%s)",
                save_path, train_scores.tolist(), val_scores.tolist(), test_scores.tolist()
            )
            if use_wandb:
                wandb.summary["best_val_scores"] = best_val_scores.tolist()

        if (epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            logger.info(
                "Epoch %d/%d | TrainScores=%s ValScores=%s TestScores=%s | TrainLoss=%s",
                epoch + 1, num_epochs,
                train_scores.tolist(),
                val_scores.tolist(),
                test_scores.tolist(),
                [loss.item() for loss in train_losses]
            )

    save(save_path, None, None, None, logs)

    if use_wandb:
        wandb.finish()

def build_merged_model(models, max_input_dim: int) -> torch.nn.Module:
    model_type: list[str] = [metadata['model_type'] for _, metadata in models]
    hidden_dims: list[int] = [metadata['hidden_dim'] for _, metadata in models]
    num_layers: list[int] = [int(metadata.get('num_layers', 2)) for _, metadata in models]
    assert all([m == model_type[0] for m in model_type])
    assert all([dim == hidden_dims[0] for dim in hidden_dims])
    assert all([nl == num_layers[0] for nl in num_layers])

    hidden_dim = hidden_dims[0]
    return utils.build_model(
        model_name=model_type[0],
        input_dim=max_input_dim,
        num_labels=None,
        device=utils.get_device(),
        hidden_dim=hidden_dim,
        num_layers=num_layers[0],
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="apply gnn merge on models")
    parser.add_argument("--model-path", action="append", type=Path, help="path for models to be merged")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="model seed")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WD, help="weight decay")
    parser.add_argument("--save-path", type=Path, help="where to save the merged model")
    parser.add_argument("--mse-loss-weight", type=float, default=1.0, help='how much to weigh mse loss?')
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.0, help='how much to weigh contrastive loss?')
    parser.add_argument("--kl-loss-weight", type=float, default=0.0, help="how much to weigh KL divergence loss?")
    parser.add_argument(
        "--kl-temperature",
        type=float,
        default=DEFAULT_KL_TEMPERATURE,
        help="temperature for KL divergence over feature distributions",
    )
    parser.add_argument("--subsample-ratio", type=float, default=1.0, help="how many of each class to (sub)sample?")
    parser.add_argument(
        "--subsample-mode",
        type=str,
        default=SubsampleMode.GT_CLASS_STRATIFIED_RANDOM.value,
        choices=[m.value for m in SubsampleMode],
        help="subsampling strategy for selecting train nodes",
    )
    parser.add_argument(
        "--soft-label-temperature",
        type=float,
        default=DEFAULT_SOFT_LABEL_TEMPERATURE,
        help="temperature for parent_soft_labels subsampling (lower favors high-confidence nodes)",
    )
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_EPOCHS, help="number of merge training epochs")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="number of GNN conv layers in the source checkpoints; inferred from metadata when omitted",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default=DEFAULT_TRAINING_MODE,
        choices=["layerwise", "e2e"],
        help="merge training strategy: per-layer teacher forcing or end-to-end final embedding matching",
    )
    # wandb
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name (omit to disable wandb)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name (optional)")
    args = parser.parse_args()
    if args.subsample_mode in {
        SubsampleMode.PARENT_SOFT_LABEL_CLASS_STRATIFIED.value,
        SubsampleMode.PARENT_ENTROPY_CLASS_STRATIFIED.value,
    } and args.soft_label_temperature <= 0:
        raise ValueError(
            "--soft-label-temperature must be > 0 for soft-label or entropy-based subsampling modes"
        )

    if args.model_path is None or args.save_path is None:
        parser.print_help()
        exit(0)

    utils.__init__randomness__(args.seed)

    device = utils.get_device()

    models = [utils.load_models(path, task=utils.Task.NodeClassification, device=device) for path in args.model_path]
    source_model_metadata = [metadata for _, metadata in models]
    source_num_layers = int(source_model_metadata[0].get("num_layers", 2))
    if args.num_layers is not None and args.num_layers != source_num_layers:
        raise ValueError(
            f"Requested num_layers={args.num_layers} does not match source checkpoints ({source_num_layers})"
        )
    if any(int(meta.get("num_layers", 2)) != source_num_layers for meta in source_model_metadata):
        raise ValueError("All source models must have the same num_layers")

    dataset_names: list[str] = [metadata['dataset'] for metadata in source_model_metadata]
    datasets: dict[str, Any] = {
        ds: utils.load_dataset(Path('artifacts/datasets') / (ds + ".pt"))
        for ds in set(dataset_names)
    }

    input_dims = {ds: datasets[ds][-1] for ds in set(dataset_names)}
    max_input_dim = max(input_dims.values())
    logger.info(f"Input dimensions: {input_dims}")
    logger.info(f"Maximum input dimension: {max_input_dim}")

    split_mode = resolve_split_mode(source_model_metadata)
    logger.info("Split mode: %s", split_mode)

    base_dataset_objs = {}
    for ds in set(dataset_names):
        data = datasets[ds][0].to(device)
        data = pad_dataset_features(data, max_input_dim)
        base_dataset_objs[ds] = data
        base_nodes, base_edges = graph_size(data)
        logger.info("Base graph | dataset=%s nodes=%d edges=%d", ds, base_nodes, base_edges)

    logger.info("Padding source models to accept max input dimension...")
    padded_models: list[tuple[torch.nn.Module, dict[str, Any]]] = []
    for (model, metadata) in models:
        current_input_dim = metadata['input_dim']
        if current_input_dim < max_input_dim:
            model = pad_model_first_layer(model, current_input_dim, max_input_dim)
        padded_models.append((model, dict(metadata)))

    models = padded_models

    all_masks = {}
    dataset_objs: dict[str, Any] = {}
    model_masks: list[utils.MaskType] = []
    final_models: list[tuple[torch.nn.Module, dict[str, Any]]] = []
    for model_idx, (model, metadata) in enumerate(models):
        dataset_name = metadata['dataset']
        if is_link_prediction(metadata):
            split_seed = int(metadata.get("link_split_seed", metadata.get("seed", args.seed)))
            val_ratio = float(metadata.get("link_val_ratio", 0.05))
            test_ratio = float(metadata.get("link_test_ratio", 0.1))
            train_data, val_data, test_data = task_evaluation.make_link_split(
                base_dataset_objs[dataset_name],
                split_seed,
                val_ratio,
                test_ratio,
            )
            dataset_key = f"{dataset_name}__lp_train__model{model_idx}"
            val_dataset_key = f"{dataset_name}__lp_val__model{model_idx}"
            test_dataset_key = f"{dataset_name}__lp_test__model{model_idx}"
            dataset_objs[dataset_key] = train_data.to(device)
            dataset_objs[val_dataset_key] = val_data.to(device)
            dataset_objs[test_dataset_key] = test_data.to(device)

            metadata["link_val_dataset_key"] = val_dataset_key
            metadata["link_test_dataset_key"] = test_dataset_key
            num_nodes = int(dataset_objs[dataset_key].num_nodes)
            split_mask = (
                torch.ones(num_nodes, dtype=torch.bool, device=device),
                torch.ones(num_nodes, dtype=torch.bool, device=device),
                torch.ones(num_nodes, dtype=torch.bool, device=device),
            )
        else:
            if dataset_name not in all_masks:
                masks = utils.make_label_masks(*(datasets[dataset_name][:-1]), num_classes=metadata['num_classes'])
                all_masks[dataset_name] = masks
            base_dataset_objs[dataset_name].y = datasets[dataset_name][0].y.to(device)
            split_mask = all_masks[dataset_name][metadata['chosen_class']]
            match split_mode:
                case 'transductive':
                    dataset_key = dataset_name
                    if dataset_key not in dataset_objs:
                        dataset_objs[dataset_key] = base_dataset_objs[dataset_name]
                case 'inductive':
                    dataset_key = f"{dataset_name}__class{metadata['chosen_class']}__model{model_idx}"
                    split_data, split_mask = make_inductive_subgraph(
                        base_dataset_objs[dataset_name],
                        split_mask,
                    )
                    dataset_objs[dataset_key] = split_data
                case _:
                    raise ValueError(f"Unknown split mode for node classification: {split_mode}")

        used_nodes, used_edges = graph_size(dataset_objs[dataset_key])
        logger.info(
            "Graph used for merge | dataset=%s task=%s class=%d mode=%s nodes=%d edges=%d",
            dataset_name,
            metadata_task(metadata),
            int(metadata.get("chosen_class", 0)),
            split_mode,
            used_nodes,
            used_edges,
        )

        train_mask, val_mask, test_mask = split_mask
        model_masks.append((
            train_mask.to(device),
            val_mask.to(device),
            test_mask.to(device),
        ))

        metadata["dataset_key"] = dataset_key
        metadata["split_mode"] = split_mode
        final_models.append((model, metadata))

    models = final_models

    merged_model = build_merged_model(models, max_input_dim)

    merge_model(
        merged_model,
        args.save_path,
        models,
        dataset_objs,
        model_masks,
        subsample_ratio=args.subsample_ratio,
        subsample_mode=args.subsample_mode,
        soft_label_temperature=args.soft_label_temperature,
        mse_loss_weight=args.mse_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        kl_loss_weight=args.kl_loss_weight,
        kl_temperature=args.kl_temperature,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_epochs=args.num_epochs,
        training_mode=args.training_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
