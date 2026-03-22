import argparse
import collections
from functools import partial
import json
import logging
from pathlib import Path
from typing import Iterable, Any, Optional

import torch
import torch_geometric
import wandb

import utils

DEFAULT_SEED = 42
DEFAULT_LR = 5e-2
DEFAULT_WD = 0.
DEFAULT_EPOCHS = 1000
DEFAULT_TRAINING_MODE = "layerwise"


# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gnn-merge")


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

    backbone = getattr(model, backbone_attr)

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


def save(
        path: Path,
        merged_model: Optional[torch.nn.Module]=None,
        models: Optional[list[torch.nn.Module]]=None,
        metadata: Optional[dict[str, Any]]=None,
        logs: Optional[dict]=None
):
    path.mkdir(parents=True, exist_ok=True)
    if merged_model is not None:
        torch.save(merged_model.state_dict(), path / "model.pt")
    if models is not None:
        for i, model in enumerate(models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")
    if metadata is not None:
        metadata_file = path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
    if logs is not None:
        log_file = path / "logs.json"
        log_file.write_text(json.dumps(logs, indent=2))


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
        parent = getattr(model, backbone_attr) if backbone_attr is not None else model

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

def siglip_loss(
    merged_model_outputs,
    model_output,
    masks,
    temperature: float = 1.0,
    normalize: bool = True,
):
    losses = []

    for merged, orig, mask in zip(merged_model_outputs, model_output, masks):

        sel = mask[0]

        z1 = merged[sel]
        z2 = orig[sel]

        n = z1.shape[0]
        if n == 0:
            continue

        if normalize:
            z1 = torch.nn.functional.normalize(z1, dim=1)
            z2 = torch.nn.functional.normalize(z2, dim=1)

        logits = torch.matmul(z1, z2.T) / temperature

        targets = 2 * torch.eye(n, device=logits.device) - torch.ones(n, device=logits.device)

        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            # + torch.nn.functional.binary_cross_entropy_with_logits(logits.T, targets)
        ) # / 2

        losses.append(loss)

    return torch.mean(torch.stack(losses))

def subsample_train_mask_by_class(
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
):
    if ratio >= 1.0:
        return train_mask

    new_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    classes = labels.unique()

    for c in classes:
        idx = (labels == c) & train_mask
        idx = idx.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        k = max(1, int(idx.numel() * ratio))
        perm = idx[torch.randperm(idx.numel(), generator=generator, device=generator.device)]
        chosen = perm[:k]
        new_mask[chosen] = True
    return new_mask

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
):
    merged_model.train()
    optimizer.zero_grad()
    losses = []
    mse_losses = []
    contrastive_losses = []

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

    torch.stack(losses).sum().backward()
    grad_norms = compute_grad_norms(merged_model)
    optimizer.step()
    return losses, grad_norms, mse_losses, contrastive_losses

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
):
    merged_model.train()
    optimizer.zero_grad()

    merged_outputs = [merged_model(datasets[meta['dataset']]) for meta in metadata]
    loss = criterion(merged_outputs, teacher_outputs, masks)
    loss.backward()

    grad_norms = compute_grad_norms(merged_model)
    optimizer.step()

    losses = [loss]
    mse_losses = []
    contrastive_losses = []
    if mse_loss_fn is not None:
        with torch.no_grad():
            mse_losses.append(mse_loss_fn(merged_outputs, teacher_outputs, masks).item())
    if contrastive_loss_fn is not None:
        with torch.no_grad():
            contrastive_losses.append(contrastive_loss_fn(merged_outputs, teacher_outputs, masks).item())

    return losses, grad_norms, mse_losses, contrastive_losses

def evaluate(
        merged_model,
        models,
        metadata,
        datasets,
        masks
):
    merged_model.eval()

    train_accs = []
    val_accs = []
    test_accs = []

    for i, (model, meta, mask) in enumerate(zip(models, metadata, masks)):
        merged_output = merged_model(datasets[meta['dataset']])
        predictions = model.mlp(merged_output).argmax(dim=1)
        labels = datasets[meta['dataset']].y
        correct = (predictions == labels).float()

        train_mask, val_mask, test_mask = mask

        train_acc = correct[train_mask].sum() / train_mask.sum().clamp(min=1)
        val_acc = correct[val_mask].sum() / val_mask.sum().clamp(min=1)
        test_acc = correct[test_mask].sum() / test_mask.sum().clamp(min=1)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    train_accs = torch.tensor(train_accs, device=utils.get_device())
    val_accs = torch.tensor(val_accs, device=utils.get_device())
    test_accs = torch.tensor(test_accs, device=utils.get_device())

    return train_accs, val_accs, test_accs

def merge_model(
        merged_model: torch.nn.Module,
        save_path: Path,
        models: list[tuple[torch.nn.Module, dict]],
        datasets,
        masks,
        subsample_ratio: float,
        mse_loss_weight: float,
        contrastive_loss_weight: float,
        learning_rate: float,
        weight_decay: float,
        seed: int,
        num_epochs: int = 1000,
        training_mode: str = DEFAULT_TRAINING_MODE,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
):
    source_metadata: list[dict] = [metadata for _, metadata in models]
    models: list[torch.nn.Module] = [model.eval() for model, _ in models]
    teacher_outputs = [model.backbone(datasets[metadata['dataset']]).detach() for model, metadata in zip(models, source_metadata)]
    model_outputs = []
    model_inputs = []
    if training_mode == "layerwise":
        model_hooks, model_outputs, model_inputs = register_hooks(models, hook_fn)
        _ = [model.backbone(datasets[metadata['dataset']]) for model, metadata in zip(models, source_metadata)]
    elif training_mode != "e2e":
        raise ValueError(f"Unknown training mode: {training_mode}")
    optimizer = torch.optim.Adam(merged_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    mse_loss = lambda merged_model_outputs, model_output, masks: torch.sum(
        torch.stack([
            torch.nn.MSELoss(reduction='mean')(merged[mask[0]], orig[mask[0]])
            for merged, orig, mask in zip(merged_model_outputs, model_output, masks)
        ])
    )

    match (mse_loss_weight, contrastive_loss_weight):
        case (0, w):
            criterion = siglip_loss
        case (w, 0):
            criterion = mse_loss
        case (w_mse, w_cl):
            criterion = lambda merged_model_outputs, model_output, masks: (
                w_mse * mse_loss(merged_model_outputs, model_output, masks)
                + w_cl * siglip_loss(merged_model_outputs, model_output, masks)
            )

    metadata = {
        "weight_decay": weight_decay,
        "lr": learning_rate,
        "epochs": num_epochs,
        "seed": seed,
        "model_type": source_metadata[0]['model_type'],
        "input_dim": source_metadata[0]['input_dim'],
        "hidden_dim": source_metadata[0]['hidden_dim'],
        "mse_loss_weight": mse_loss_weight,
        "contrastive_loss_weight": contrastive_loss_weight,
        "subsample_ratio": subsample_ratio,
        "training_mode": training_mode,
        "source_models": source_metadata
    }

    logs = collections.defaultdict(list)
    save(save_path, None, models, metadata, None)
    best_val_accs = torch.tensor([0. for _ in models], device=utils.get_device())

    labels = [datasets[meta['dataset']].y for meta in source_metadata]

    gen = torch.Generator(device=utils.get_device())
    gen.manual_seed(DEFAULT_SEED)

    subsampled_masks = []
    for (train_mask, val_mask, test_mask), y in zip(masks, labels):
        sub_train = subsample_train_mask_by_class(y, train_mask, subsample_ratio, gen)
        subsampled_masks.append((sub_train, val_mask, test_mask))

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
            f"{meta['dataset']}/{meta['chosen_class']}/{meta['num_classes']}"
            for meta in source_metadata
        ]
        layer_names = list(model_outputs[0].keys()) if training_mode == "layerwise" else ["final"]
        logger.info("wandb run initialised — prefixes: %s", model_prefixes)

    logger.info("starting model merging...")
    for epoch in range(num_epochs):
        if training_mode == "layerwise":
            train_losses, grad_norms, mse_losses, contrastive_losses = train(
                merged_model, model_outputs, model_inputs, subsampled_masks, optimizer, criterion,
                contrastive_loss_fn=siglip_loss if contrastive_loss_weight > 0 else None,
                mse_loss_fn=mse_loss if mse_loss_weight > 0 else None)
        else:
            train_losses, grad_norms, mse_losses, contrastive_losses = train_e2e(
                merged_model, teacher_outputs, datasets, source_metadata, subsampled_masks, optimizer, criterion,
                contrastive_loss_fn=siglip_loss if contrastive_loss_weight > 0 else None,
                mse_loss_fn=mse_loss if mse_loss_weight > 0 else None)
        train_accs, val_accs, test_accs = evaluate(merged_model, models, source_metadata, datasets, masks)

        # ---- internal logs (unchanged) ----
        for i, loss in enumerate(train_losses):
            logs[f"train_loss_{i}"].append(loss.item())
        if mse_loss_weight > 0:
            for i, loss in enumerate(mse_losses):
                logs[f"train_mse_loss_{i}"].append(loss)
        if contrastive_loss_weight > 0:
            for i, loss in enumerate(contrastive_losses):
                logs[f"train_contrastive_loss_{i}"].append(loss)
        for i, train_acc in enumerate(train_accs):
            logs[f"train_acc_{i}"].append(train_acc.item())
        for i, val_acc in enumerate(val_accs):
            logs[f"val_acc_{i}"].append(val_acc.item())
        for i, test_acc in enumerate(test_accs):
            logs[f"test_acc_{i}"].append(test_acc.item())
        logs["grad_norm_total"].append(grad_norms["total"])

        # ---- wandb logging ----
        if use_wandb:
            wb_log: dict[str, float] = {}

            # Per-model accuracy metrics, namespaced by dataset/split/total
            for i, prefix in enumerate(model_prefixes):
                wb_log[f"{prefix}/train_acc"] = train_accs[i].item()
                wb_log[f"{prefix}/val_acc"]   = val_accs[i].item()
                wb_log[f"{prefix}/test_acc"]  = test_accs[i].item()

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
        if torch.all(val_accs > best_val_accs):
            best_val_accs = val_accs.clone()
            save(save_path, merged_model, None, None, logs)
            logger.info(
                "Saved best checkpoint to %s (train_acc=%s, val_acc=%s, test_acc=%s)",
                save_path, train_accs.tolist(), val_accs.tolist(), test_accs.tolist()
            )
            if use_wandb:
                wandb.summary["best_val_accs"] = best_val_accs.tolist()

        if (epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            logger.info(
                "Epoch %d/%d | TrainAcc=%s ValAcc=%s TestAcc=%s | TrainLoss=%s",
                epoch + 1, num_epochs,
                train_accs.tolist(),
                val_accs.tolist(),
                test_accs.tolist(),
                [loss.item() for loss in train_losses]
            )

    save(save_path, None, None, None, logs)

    if use_wandb:
        wandb.finish()

def build_merged_model(models, max_input_dim: int) -> torch.nn.Module:
    model_type: list[str] = [metadata['model_type'] for _, metadata in models]
    hidden_dims: list[int] = [metadata['hidden_dim'] for _, metadata in models]
    assert all([m == model_type[0] for m in model_type])
    assert all([dim == hidden_dims[0] for dim in hidden_dims])

    hidden_dim = hidden_dims[0]
    return utils.build_model(
        model_name=model_type[0],
        input_dim=max_input_dim,
        num_labels=None,
        device=utils.get_device(),
        hidden_dim=hidden_dim
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="apply gnn merge on models")
    parser.add_argument("--model-path", action="append", type=Path, help="path for models to be merged")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="model seed")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WD, help="weight decay")
    parser.add_argument("--save-path", type=Path, help="where to save the merged model")
    parser.add_argument("--mse-loss-weight", type=float, help='how much to weigh mse loss?')
    parser.add_argument("--contrastive-loss-weight", type=float, help='how much to weigh contrastive loss?')
    parser.add_argument("--subsample-ratio", type=float, default=1.0, help="how many of each class to (sub)sample?")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_EPOCHS, help="number of merge training epochs")
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

    if args.model_path is None or args.save_path is None:
        parser.print_help()
        exit(0)

    utils.__init__randomness__(args.seed)

    device = utils.get_device()

    models = [utils.load_models(path, task=utils.Task.NodeClassification, device=device) for path in args.model_path]
    source_model_metadata = [metadata for _, metadata in models]

    dataset_names: list[str] = [metadata['dataset'] for metadata in source_model_metadata]
    datasets: dict[str, Any] = {
        ds: utils.load_dataset(Path('artifacts/datasets') / (ds + ".pt"))
        for ds in set(dataset_names)
    }

    input_dims = {ds: datasets[ds][-1] for ds in set(dataset_names)}
    max_input_dim = max(input_dims.values())
    logger.info(f"Input dimensions: {input_dims}")
    logger.info(f"Maximum input dimension: {max_input_dim}")

    dataset_objs = {}
    for ds in set(dataset_names):
        data = datasets[ds][0].to(device)
        data = pad_dataset_features(data, max_input_dim)
        dataset_objs[ds] = data

    logger.info("Padding source models to accept max input dimension...")
    padded_models = []
    for (model, metadata) in models:
        current_input_dim = metadata['input_dim']
        if current_input_dim < max_input_dim:
            model = pad_model_first_layer(model, current_input_dim, max_input_dim)
        padded_models.append((model, metadata))

    models = padded_models

    all_masks = {}
    model_masks = []
    for metadata in source_model_metadata:
        dataset_name = metadata['dataset']
        if dataset_name not in all_masks:
            masks = utils.make_label_masks(*(datasets[dataset_name][:-1]), num_classes=metadata['num_classes'])
            all_masks[dataset_name] = masks
        model_masks.append(all_masks[dataset_name][metadata['chosen_class']])

    model_masks = [tuple(m.to(device) for m in mask) for mask in model_masks]

    merged_model = build_merged_model(models, max_input_dim)

    merge_model(
        merged_model,
        args.save_path,
        models,
        dataset_objs,
        model_masks,
        subsample_ratio=args.subsample_ratio,
        mse_loss_weight=args.mse_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_epochs=args.num_epochs,
        training_mode=args.training_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
