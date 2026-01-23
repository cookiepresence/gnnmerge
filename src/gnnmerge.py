import argparse
import collections
from functools import partial
import json
import logging
from pathlib import Path
from typing import Iterable, Any, Optional

import torch
import torch_geometric

import utils

DEFAULT_SEED = 42
DEFAULT_LR = 5e-2
DEFAULT_WD = 0.


# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",

datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gnn-merge")


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
    outs[layer_name] = output
    ins[layer_name] = input

def register_hooks(
    models: Iterable,
    hook_fn,
    backbone_attr: str = "backbone",
) -> tuple[list[list], list[dict], list[dict]]:
    """
    Register forward hooks on every layer in each model's backbone.

    Args:
        models: iterable of model objects (e.g. [model1, model2, ...]).
        hook_fn: callable used as the hook; will be partially applied with outs, ins, layer_name.
                 Signature after partial must match PyTorch forward hook: (module, input, output).
        backbone_attr: attribute name pointing to the backbone module (default "backbone").
        include_containers: if False (default), register only on leaf modules (modules with no child modules).
                            if True, register on every module returned by named_modules().
        include_root: whether to include the root backbone module itself (named_modules yields root with name '').

    Returns:
        (hooks, outs, ins)
        - hooks: list (per model) of lists of hook handles.
        - outs:  list (per model) of lists where hook_fn can append outputs.
        - ins:   list (per model) of lists where hook_fn can append inputs.
    Notes:
        - No try/except; missing backbone_attr or other attribute lookups will raise.
        - The layer_name passed to hook_fn will be prefixed with "m{index}:" to keep names unique across models.
    """
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
            # skip if it is the root module or a nested child module
            # we only want immiediate children!!
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


def train(
        merged_model: torch.nn.Module,
        model_outputs: list[dict],
        model_inputs: list[dict],
        masks: list[utils.MaskType],
        optimizer,
        criterion
):
    merged_model.train()
    optimizer.zero_grad()

    losses = []

    # assumption: all models have the same keys
    for layer in model_outputs[0].keys():
        model_output = [outputs[layer] for outputs in model_outputs]
        model_input = [inputs[layer] for inputs in model_inputs]

        # print(model_output)
        # print(model_input)
        
        merged_model_layer = getattr(merged_model, layer)
        merged_model_outputs = [merged_model_layer(*inp) for inp in model_input]
        loss = torch.sum(
            torch.stack([
                criterion(merged[mask[0]], orig[mask[0]])
                for merged, orig, mask in zip(model_output, merged_model_outputs, masks)
            ])
        )
        losses.append(loss)

    torch.stack(losses).sum().backward(retain_graph=True)
    optimizer.step()
    return losses

def evaluate(
        merged_model,
        models,
        metadata,
        datasets,
        masks
):
    merged_model.eval()

    merged_output = [merged_model(datasets[meta['dataset']]) for meta in metadata]

    predictions = [model.mlp(out).argmax(dim=1) for model, out in zip(models, merged_output)]
    predictions = torch.stack(predictions)
    labels = torch.stack([datasets[meta['dataset']].y for meta in metadata])

    # predictions shape: [2, xxx]
    # labels shape: [2, xxx]
    # masks shape: [num_models, 3, xxx]
    
    # stack the masks so that we can index them comfortably
    masks = torch.cat([torch.stack(m).unsqueeze(1) for m in masks], dim=1).to(utils.get_device())
    # sks = masks.view(3, len(models), -1) # [3, num_models, xxx]
    correct = (predictions == labels).float().unsqueeze(0)     # [1, num_models, num_samples]
    
    masked_correct = (correct * masks).sum(dim=2)  # [3, num_models]
    mask_sizes = masks.sum(dim=2)  # [3, num_models]
    accuracies = masked_correct / mask_sizes.clamp(min=1)

    train_accs = accuracies[0]
    val_accs = accuracies[1]
    test_accs = accuracies[2]
    
    return train_accs, val_accs, test_accs

def merge_model(
        merged_model: torch.nn.Module,
        save_path: Path,
        models: list[tuple[torch.nn.Module, dict]],
        datasets,
        masks,
        learning_rate: float,
        weight_decay: float,
        seed: int,
        num_epochs: int = 1000
):
    source_metadata: list[dict] = [metadata for _, metadata in models]
    models: list[torch.nn.Module] = [model.eval() for model, _ in models]
    model_hooks, model_outputs, model_inputs = register_hooks(models, hook_fn)
    _ = [model.backbone(datasets[metadata['dataset']]) for model, metadata in zip(models, source_metadata)]
    optimizer = torch.optim.Adam(merged_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss(reduction='mean')
    metadata = {
        "weight_decay": weight_decay,
        "lr": learning_rate,
        "epochs": num_epochs,
        "seed": seed,
        "model_type": source_metadata[0]['model_type'],
        "input_dim": source_metadata[0]['input_dim'],
        "hidden_dim": source_metadata[0]['hidden_dim'],
        "source_models": source_metadata
    }
    logs = collections.defaultdict(list)
    save(save_path, None, models, metadata, None)
    best_val_accs = torch.tensor([0. for _ in models], device=utils.get_device())
    
    logger.info("starting model merging...")
    for epoch in range(num_epochs):
        train_losses = train(merged_model, model_outputs, model_inputs, masks, optimizer, criterion)
        train_accs, val_accs, test_accs = evaluate(merged_model, models, source_metadata, datasets, masks)
        
        # Log to collections - convert tensors to Python lists/scalars
        for i, loss in enumerate(train_losses):
            logs[f"train_loss_{i}"].append(loss.item())
        for i, train_acc in enumerate(train_accs):
            logs[f"train_acc_{i}"].append(train_acc.item())
        for i, val_acc in enumerate(val_accs):
            logs[f"val_acc_{i}"].append(val_acc.item())
        for i, test_acc in enumerate(test_accs):
            logs[f"test_acc_{i}"].append(test_acc.item())
        
        # Check if current validation accuracies are better (all should be better)
        if torch.all(val_accs > best_val_accs):
            best_val_accs = val_accs.clone()
            save(save_path, merged_model, None, None, logs)
            logger.info("Saved best checkpoint for the merged model to %s (val_acc=%s, test_acc=%s)", 
                       save_path, 
                       val_accs.tolist(), 
                       test_accs.tolist())
        
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

def build_merged_model(models, datasets) -> torch.nn.Module:
    model_type: list[str] = [metadata['model_type'] for _, metadata in models]
    hidden_dims: list[int] = [metadata['hidden_dim'] for _, metadata in models]
    # all models are the same type
    assert all([m == model_type[0] for m in model_type])
    # all models have the same input dim
    assert all([dim == hidden_dims[0] for dim in hidden_dims])

    # stored as a tuple, hence we get it this way
    input_dim = datasets[list(datasets.keys())[0]][-1]
    hidden_dim = hidden_dims[0]
    return utils.build_model(
        model_name=model_type[0],
        input_dim=input_dim,
        # since we only care about the backbone
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
    args = parser.parse_args()

    if args.model_path is None or args.save_path is None:
        parser.print_help()
        exit(0)

    utils.__init__randomness__(args.seed)

    device = utils.get_device()
    
    # currently, we only care about in-domain merging
    # not sure how tasks training on diff datasets have the same input dim
    models = [utils.load_models(path, task=utils.Task.NodeClassification, device=device) for path in args.model_path]
    source_model_metadata = [metadata for _, metadata in models]
    
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

    merged_model = build_merged_model(models, datasets)

    merge_model(
        merged_model,
        args.save_path,
        models,
        dataset_objs,
        model_masks,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed
    )
