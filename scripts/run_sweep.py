#!/usr/bin/env python3
"""Unified standalone sweep runner.

Subcommands:
- merge: sweep src/gnnmerge.py over pairings + hyperparameter grids
- train: sweep src/train_node_classification.py over training grids

No dependency on scripts/prepare_merge_experiments.py or scripts/run_single_merge_experiment.py.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DIR_RE = re.compile(
    r"^(?P<arch>gcn|sage|gat)_(?P<dataset>.+?)_(?:model_)?(?P<split>\d+)-(?P<total>\d+)_(?P<dim>\d+)dims_(?P<depth>\d+)l_(?P<type>ind|trans)$"
)


@dataclass(frozen=True)
class ModelDir:
    path: Path
    arch: str
    dataset: str
    split: int
    total: int
    dim: int
    depth: int
    type: str


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    pair_label: str
    domain: str
    arch: str
    datasets: tuple[str, ...]
    model_paths: tuple[str, ...]


def fmt_float(value: float) -> str:
    s = f"{value:.8g}"
    return s.replace("-", "m").replace(".", "p")


def stable_id(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def parse_float_list(values: list[str]) -> list[float]:
    return [float(v) for v in values]


def parse_int_list(values: list[str]) -> list[int]:
    return [int(v) for v in values]


def parse_coeff_grid(value: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid coeff entry '{item}', expected mse:contrastive")
        out.append((float(parts[0]), float(parts[1])))
    if not out:
        raise ValueError("Coefficient grid is empty")
    return out


def parse_optional_set(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {v for v in (s.strip() for s in values) if v}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def discover_models(model_root: Path, hidden_dim: int, arches: set[str], datasets: set[str] | None) -> list[ModelDir]:
    found: list[ModelDir] = []
    for entry in sorted(model_root.iterdir()):
        if not entry.is_dir():
            continue
        m = DIR_RE.match(entry.name)
        if m is None:
            continue
        arch = m.group("arch")
        dataset = m.group("dataset")
        dim = int(m.group("dim"))
        depth = int(m.group("depth"))
        type = m.group("type")
        if arch not in arches or dim != hidden_dim:
            continue
        if datasets is not None and dataset not in datasets:
            continue
        found.append(
            ModelDir(
                path=entry,
                arch=arch,
                dataset=dataset,
                split=int(m.group("split")),
                total=int(m.group("total")),
                dim=dim,
                depth=depth,
                type=type
            )
        )
    return found


def build_in_domain_pairs(models: list[ModelDir]) -> list[PairSpec]:
    groups: dict[tuple[str, str, int], dict[int, ModelDir]] = defaultdict(dict)
    for m in models:
        groups[(m.arch, m.dataset, m.total)][m.split] = m

    out: list[PairSpec] = []
    for (arch, dataset, total), split_map in sorted(groups.items()):
        if total < 2:
            continue
        expected = set(range(total))
        if set(split_map) != expected:
            continue
        model_paths = tuple(str(split_map[i].path) for i in sorted(split_map))
        pair_label = f"{arch}__{dataset}__splits0to{total - 1}"
        out.append(
            PairSpec(
                pair_id=stable_id({"domain": "in_domain", "arch": arch, "dataset": dataset, "paths": model_paths}),
                pair_label=pair_label,
                domain="in_domain",
                arch=arch,
                datasets=(dataset,),
                model_paths=model_paths,
            )
        )
    return out


def build_cross_pairs(models: list[ModelDir], mode: str) -> list[PairSpec]:
    out: list[PairSpec] = []
    if mode == "full_train":
        by_arch: dict[str, dict[str, ModelDir]] = defaultdict(dict)
        for m in models:
            if m.total == 1 and m.split == 0:
                by_arch[m.arch][m.dataset] = m

        for arch, ds_map in sorted(by_arch.items()):
            for left, right in itertools.combinations(sorted(ds_map), 2):
                model_paths = (str(ds_map[left].path), str(ds_map[right].path))
                pair_label = f"{arch}__{left}__{right}__fulltrain"
                out.append(
                    PairSpec(
                        pair_id=stable_id({"domain": "cross_domain", "mode": mode, "arch": arch, "left": left, "right": right, "paths": model_paths}),
                        pair_label=pair_label,
                        domain="cross_domain",
                        arch=arch,
                        datasets=(left, right),
                        model_paths=model_paths,
                    )
                )
        return out

    by_key: dict[tuple[str, int, int], dict[str, ModelDir]] = defaultdict(dict)
    for m in models:
        by_key[(m.arch, m.total, m.split)][m.dataset] = m

    for (arch, total, split), ds_map in sorted(by_key.items()):
        for left, right in itertools.combinations(sorted(ds_map), 2):
            model_paths = (str(ds_map[left].path), str(ds_map[right].path))
            pair_label = f"{arch}__{left}__{right}__split{split}of{total}"
            out.append(
                PairSpec(
                    pair_id=stable_id({"domain": "cross_domain", "mode": mode, "arch": arch, "left": left, "right": right, "split": split, "total": total, "paths": model_paths}),
                    pair_label=pair_label,
                    domain="cross_domain",
                    arch=arch,
                    datasets=(left, right),
                    model_paths=model_paths,
                )
            )
    return out


def build_merge_rows(args: argparse.Namespace) -> list[dict]:
    arches = set(args.arches)
    dataset_filter = parse_optional_set(args.datasets)
    models = discover_models(args.model_root, args.hidden_dim, arches, dataset_filter)
    if not models:
        raise RuntimeError("No matching model directories found")

    pairs: list[PairSpec] = []
    if "in_domain" in args.domains:
        pairs.extend(build_in_domain_pairs(models))
    if "cross_domain" in args.domains:
        pairs.extend(build_cross_pairs(models, args.cross_domain_mode))
    if not pairs:
        raise RuntimeError("No valid pairs discovered for selected domains")

    ratios = parse_float_list(args.ratios)
    seeds = parse_int_list(args.seeds)
    coeffs = parse_coeff_grid(args.coeff_grid)
    lrs = parse_float_list(args.lr_grid)
    wds = parse_float_list(args.wd_grid)
    training_modes = args.training_modes

    rows: list[dict] = []
    for pair, ratio, seed, (mse_w, cl_w), lr, wd, mode, ramp in itertools.product(
        pairs, ratios, seeds, coeffs, lrs, wds, training_modes
    ):
        if args.max_contrastive_for_ogbn_arxiv >= 0 and "ogbn_arxiv" in pair.datasets and cl_w > args.max_contrastive_for_ogbn_arxiv:
            continue

        ratio_tag = fmt_float(ratio)
        mse_tag = fmt_float(mse_w)
        cl_tag = fmt_float(cl_w)
        lr_tag = fmt_float(lr)
        wd_tag = fmt_float(wd)
        ramp_tag = fmt_float(ramp)

        save_path = (
            args.run_root
            / "tuning"
            / pair.domain
            / pair.arch
            / pair.pair_label
            / f"ratio_{ratio_tag}"
            / f"seed_{seed}"
            / f"mode_{mode}__mse_{mse_tag}__cl_{cl_tag}__lr_{lr_tag}__wd_{wd_tag}__ramp_{ramp_tag}__nl_{args.num_layers}"
        )

        run_payload = {
            "task": "merge",
            "domain": pair.domain,
            "arch": pair.arch,
            "pair_id": pair.pair_id,
            "pair_label": pair.pair_label,
            "datasets": list(pair.datasets),
            "model_paths": list(pair.model_paths),
            "seed": int(seed),
            "subsample_ratio": float(ratio),
            "mse_loss_weight": float(mse_w),
            "contrastive_loss_weight": float(cl_w),
            "lr": float(lr),
            "weight_decay": float(wd),
            "training_mode": mode,
            "num_layers": int(args.num_layers),
            "num_epochs": int(args.num_epochs),
            "save_path": str(save_path),
        }
        run_payload["run_id"] = stable_id(run_payload)

        if args.wandb_project:
            run_payload["wandb_project"] = args.wandb_project
            run_payload["wandb_run_name"] = args.wandb_name_template.format(
                domain=pair.domain,
                arch=pair.arch,
                pair_label=pair.pair_label,
                ratio=ratio_tag,
                seed=seed,
                mse=mse_tag,
                cl=cl_tag,
                lr=lr_tag,
                wd=wd_tag,
                training_mode=mode,
                ramp=ramp_tag,
                num_layers=args.num_layers,
                run_id=run_payload["run_id"],
            )

        rows.append(run_payload)

    return rows


def build_train_rows(args: argparse.Namespace) -> list[dict]:
    datasets = args.datasets
    models = args.models
    chosen_classes = parse_int_list(args.chosen_classes)
    seeds = parse_int_list(args.seeds)
    hidden_dims = parse_int_list(args.hidden_dims)
    lrs = parse_float_list(args.lr_grid)
    wds = parse_float_list(args.wd_grid)

    rows: list[dict] = []
    for dataset, model, chosen_class, seed, hidden_dim, lr, wd in itertools.product(
        datasets, models, chosen_classes, seeds, hidden_dims, lrs, wds
    ):
        if chosen_class < 0 or chosen_class >= args.num_classes:
            continue

        data_path = args.data_path_template.format(dataset=dataset)
        save_path = args.save_path_template.format(
            model=model,
            dataset=dataset,
            chosen_class=chosen_class,
            num_classes=args.num_classes,
            hidden_dim=hidden_dim,
            seed=seed,
            lr=fmt_float(lr),
            wd=fmt_float(wd),
            num_layers=args.num_layers,
        )

        run_payload = {
            "task": "train",
            "dataset": dataset,
            "model": model,
            "data_path": data_path,
            "save_path": save_path,
            "num_classes": int(args.num_classes),
            "chosen_class": int(chosen_class),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(args.num_layers),
            "num_epochs": int(args.num_epochs),
            "seed": int(seed),
            "lr": float(lr),
            "weight_decay": float(wd),
            "transductive": bool(args.transductive),
        }
        run_payload["run_id"] = stable_id(run_payload)

        if args.wandb_project:
            run_payload["wandb_project"] = args.wandb_project
            run_payload["wandb_name"] = args.wandb_name_template.format(
                model=model,
                dataset=dataset,
                chosen_class=chosen_class,
                num_classes=args.num_classes,
                hidden_dim=hidden_dim,
                seed=seed,
                lr=fmt_float(lr),
                wd=fmt_float(wd),
                num_layers=args.num_layers,
                run_id=run_payload["run_id"],
            )

        rows.append(run_payload)

    return rows


def build_merge_command(row: dict) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python3",
        "src/gnnmerge.py",
        "--seed",
        str(row["seed"]),
        "--lr",
        str(row["lr"]),
        "--weight-decay",
        str(row["weight_decay"]),
        "--save-path",
        str(row["save_path"]),
        "--subsample-ratio",
        str(row["subsample_ratio"]),
        "--mse-loss-weight",
        str(row["mse_loss_weight"]),
        "--contrastive-loss-weight",
        str(row["contrastive_loss_weight"]),
        "--num-epochs",
        str(row["num_epochs"]),
        "--num-layers",
        str(row["num_layers"]),
        "--training-mode",
        str(row["training_mode"]),
        "--contrastive-ramp-fraction",
        str(row["contrastive_ramp_fraction"]),
    ]
    for p in row["model_paths"]:
        cmd.extend(["--model-path", str(p)])
    if row.get("wandb_project"):
        cmd.extend(["--wandb-project", row["wandb_project"]])
        cmd.extend(["--wandb-run-name", row["wandb_run_name"]])
    return cmd


def build_train_command(row: dict) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python3",
        "src/train_node_classification.py",
        "--dataset",
        row["dataset"],
        "--model",
        row["model"],
        "--data-path",
        row["data_path"],
        "--save-path",
        row["save_path"],
        "--num-classes",
        str(row["num_classes"]),
        "--chosen-class",
        str(row["chosen_class"]),
        "--num-epochs",
        str(row["num_epochs"]),
        "--lr",
        str(row["lr"]),
        "--weight-decay",
        str(row["weight_decay"]),
        "--hidden-dim",
        str(row["hidden_dim"]),
        "--num-layers",
        str(row["num_layers"]),
        "--seed",
        str(row["seed"]),
    ]
    if row.get("transductive"):
        cmd.append("--transductive")
    if row.get("wandb_project"):
        cmd.extend(["--wandb-project", row["wandb_project"]])
        cmd.extend(["--wandb-name", row["wandb_name"]])
    else:
        # train_node_classification.py requires --wandb-name.
        cmd.extend(["--wandb-name", row["run_id"]])
    return cmd


def execute_rows(rows: list[dict], *, remote: bool, start_index: int, limit: int, stop_on_error: bool) -> int:
    if not rows:
        print("[error] no runs produced", file=sys.stderr)
        return 1

    total = len(rows)
    if start_index < 1 or start_index > total:
        print(f"[error] --start-index must be in [1, {total}]", file=sys.stderr)
        return 2

    end_index = total
    if limit > 0:
        end_index = min(total, start_index + limit - 1)

    failures = 0
    mode = "remote" if remote else "local"
    print(f"[runs] total={total} selected={start_index}..{end_index} mode={mode}")
    for idx in range(start_index, end_index + 1):
        row = rows[idx - 1]
        if row["task"] == "merge":
            base_cmd = build_merge_command(row)
        else:
            base_cmd = build_train_command(row)

        if remote:
            inner = " ".join(base_cmd)
            cmd = ["tools/remote.sh", "run", inner]
        else:
            cmd = base_cmd

        print(f"[{idx}/{total}] run_id={row['run_id']}")
        print("[cmd]", " ".join(cmd))

        code = subprocess.run(cmd).returncode
        if code != 0:
            failures += 1
            print(f"[warn] failed index={idx} code={code}", file=sys.stderr)
            if stop_on_error:
                return code

    if failures:
        print(f"[done] completed with failures={failures}", file=sys.stderr)
        return 1

    print("[done] all selected runs succeeded")
    return 0


def add_common_execution_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest-path", type=Path, required=False, default=None)
    p.add_argument("--remote", action="store_true", help="Run commands via tools/remote.sh instead of locally")
    p.add_argument("--start-index", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Write the manifest and stop without executing runs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone GNNMerge sweep runner")
    sub = p.add_subparsers(dest="task", required=True)

    p_merge = sub.add_parser("merge", help="Sweep src/gnnmerge.py")
    p_merge.add_argument("--model-root", type=Path, default=Path("artifacts/models"))
    p_merge.add_argument("--run-root", type=Path, default=Path("artifacts/merged_sweeps"))
    p_merge.add_argument("--out-dir", type=Path, default=Path("artifacts/experiment_manifests"))
    p_merge.add_argument("--hidden-dim", type=int, default=128)
    p_merge.add_argument("--arches", nargs="+", default=["gcn", "sage"])
    p_merge.add_argument("--datasets", nargs="+", default=[])
    p_merge.add_argument("--domains", nargs="+", choices=["in_domain", "cross_domain"], default=["in_domain", "cross_domain"])
    p_merge.add_argument("--cross-domain-mode", choices=["full_train", "label_split"], default="full_train")
    p_merge.add_argument("--ratios", nargs="+", default=["0.01", "0.1", "1.0"])
    p_merge.add_argument("--seeds", nargs="+", default=["101", "202", "303"])
    p_merge.add_argument("--coeff-grid", type=str, default="1:0,1:0.1,0.9:0.1,0.5:0.5")
    p_merge.add_argument("--lr-grid", nargs="+", default=["0.005", "0.01", "0.02", "0.05"])
    p_merge.add_argument("--wd-grid", nargs="+", default=["0.0", "1e-6", "1e-5", "1e-4"])
    p_merge.add_argument("--num-epochs", type=int, default=300)
    p_merge.add_argument("--num-layers", type=int, default=2)
    p_merge.add_argument("--training-modes", nargs="+", default=["layerwise"])
    p_merge.add_argument("--contrastive-ramp-fraction-grid", nargs="+", default=["0.0"])
    p_merge.add_argument("--max-contrastive-for-ogbn-arxiv", type=float, default=0.2)
    p_merge.add_argument("--wandb-project", type=str, default="gnn-merge-sweep")
    p_merge.add_argument(
        "--wandb-name-template",
        type=str,
        default="merge__{domain}__{arch}__{pair_label}__r{ratio}__seed{seed}__m{mse}__c{cl}__lr{lr}__wd{wd}__tm{training_mode}__nl{num_layers}__id{run_id}",
    )
    add_common_execution_args(p_merge)

    p_train = sub.add_parser("train", help="Sweep src/train_node_classification.py")
    p_train.add_argument("--out-dir", type=Path, default=Path("artifacts/experiment_manifests"))
    p_train.add_argument("--datasets", nargs="+", default=["amazon_computers", "amazon_photo", "cora", "ogbn_arxiv", "wikics"])
    p_train.add_argument("--models", nargs="+", default=["gcn", "sage"])
    p_train.add_argument("--num-classes", type=int, default=2)
    p_train.add_argument("--chosen-classes", nargs="+", default=["0", "1"])
    p_train.add_argument("--hidden-dims", nargs="+", default=["128"])
    p_train.add_argument("--seeds", nargs="+", default=["42"])
    p_train.add_argument("--lr-grid", nargs="+", default=["0.005"])
    p_train.add_argument("--wd-grid", nargs="+", default=["0.0005"])
    p_train.add_argument("--num-epochs", type=int, default=150)
    p_train.add_argument("--num-layers", type=int, default=2)
    p_train.add_argument("--transductive", action="store_true")
    p_train.add_argument("--data-path-template", type=str, default="artifacts/datasets/{dataset}.pt")
    p_train.add_argument(
        "--save-path-template",
        type=str,
        default="artifacts/models/{model}_{dataset}_model_{chosen_class}-{num_classes}_{hidden_dim}dims_seed{seed}_lr{lr}_wd{wd}_nl{num_layers}",
    )
    p_train.add_argument("--wandb-project", type=str, default="gnnmerge-repro")
    p_train.add_argument(
        "--wandb-name-template",
        type=str,
        default="train__{model}__{dataset}__split{chosen_class}of{num_classes}__h{hidden_dim}__seed{seed}__lr{lr}__wd{wd}__nl{num_layers}__id{run_id}",
    )
    add_common_execution_args(p_train)

    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.task == "merge":
        rows = build_merge_rows(args)
        manifest = args.manifest_path or (args.out_dir / "merge_sweep_runs.jsonl")
    else:
        rows = build_train_rows(args)
        manifest = args.manifest_path or (args.out_dir / "train_sweep_runs.jsonl")

    write_jsonl(manifest, rows)
    print(f"[manifest] wrote {len(rows)} rows to {manifest}")

    if args.dry_run:
        print("[done] dry-run complete")
        return 0

    return execute_rows(
        rows,
        remote=args.remote,
        start_index=args.start_index,
        limit=args.limit,
        stop_on_error=args.stop_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
