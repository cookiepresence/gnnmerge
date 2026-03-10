#!/usr/bin/env python3
"""
sweep.py — grid search over:
  model arch × dataset × hidden dim × subsample ratio × loss function

Paired source models are discovered automatically by scanning MODEL_ROOT
and matching directories with the naming convention:
  {arch}_{dataset}[_model]_{split}-{total}_{dim}dims
  e.g.  gcn_wikics_model_0-2_128dims  /  gcn_wikics_model_1-2_128dims

Usage:
  python sweep.py            # run everything
  python sweep.py --dry-run  # print commands without executing
"""

import argparse
import itertools
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------ config ---
WANDB_PROJECT = "gnn-merge-sweep"
MODEL_ROOT    = Path("artifacts/models")
SAVE_ROOT     = Path("artifacts/merged")

RATIOS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
LOSSES = ["mse", "contrastive"]

SEED   = 42
LR     = 5e-2
WD     = 0.0
# to save time, otherwise it will take forever
EPOCHS = 250

# Regex covers both  *_model_0-2_128dims  and  *_0-2_128dims
DIR_RE = re.compile(
    r"^(?P<arch>gcn|sage)_(?P<dataset>.+?)_(?:model_)?(?P<split>\d+)-(?P<total>\d+)_(?P<dim>\d+)dims$"
)

# ------------------------------------------------------------------ types ---
@dataclass
class ModelDir:
    path:    Path
    arch:    str
    dataset: str
    dim:     int
    split:   int
    total:   int


@dataclass
class MergeGroup:
    """Two complementary split models that will be merged together."""
    arch:    str
    dataset: str
    dim:     int
    paths:   list[Path] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.arch}__{self.dataset}__{self.dim}dims"


# --------------------------------------------------------- model discovery ---
def discover_groups(root: Path) -> list[MergeGroup]:
    """
    Scan MODEL_ROOT, parse directory names, and group by (arch, dataset, dim).
    Only groups where all expected splits [0..total-1] are present are kept.
    """
    by_group: dict[tuple, dict[int, Path]] = defaultdict(dict)

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        m = DIR_RE.match(entry.name)
        if m is None:
            print(f"[sweep] SKIP (unrecognised name): {entry.name}", file=sys.stderr)
            continue
        arch    = m.group("arch")
        dataset = m.group("dataset")
        dim     = int(m.group("dim"))
        split   = int(m.group("split"))
        total   = int(m.group("total"))

        if dim == 2:
            print(f"[sweep] SKIP (small dim): {entry.name}", file=sys.stderr)
            continue

        if dataset == 'wikics' or dataset == 'cora':
            print(f"[sweep] SKIP (garbage dataset): {entry.name}", file=sys.stderr)
            continue

        by_group[(arch, dataset, dim, total)][split] = entry

    groups: list[MergeGroup] = []
    for (arch, dataset, dim, total), split_map in sorted(by_group.items()):
        expected = set(range(total))
        if set(split_map.keys()) != expected:
            missing = expected - set(split_map.keys())
            print(
                f"[sweep] SKIP incomplete group ({arch}, {dataset}, {dim}d): "
                f"missing splits {missing}",
                file=sys.stderr,
            )
            continue
        g = MergeGroup(arch=arch, dataset=dataset, dim=dim)
        for split_idx in sorted(split_map):
            g.paths.append(split_map[split_idx])
        groups.append(g)

    return groups


# ------------------------------------------------------------------ sweep ---
def build_command(
    group:      MergeGroup,
    ratio:      float,
    loss:       str,
    save_root:  Path,
    wandb_project: str,
) -> tuple[str, Path]:
    """Return (wandb_run_name, save_path, cmd_list)."""

    ratio_tag = f"r{ratio}"
    run_name  = f"{group.arch}__{group.dataset}__{group.dim}dims__{ratio_tag}__{loss}"
    save_path = save_root / group.arch / group.dataset / f"{group.dim}dims" / ratio_tag / loss

    cmd = [
        "uv", "run", "python3", "src/gnnmerge.py",
        "--seed",            str(SEED),
        "--lr",              str(LR),
        "--weight-decay",    str(WD),
        "--save-path",       str(save_path),
        "--subsample-ratio", str(ratio),
        "--wandb-project",   wandb_project,
        "--wandb-run-name",  run_name,
    ]
    for p in group.paths:
        cmd += ["--model-path", str(p)]
    if loss == "contrastive":
        cmd.append("--contrastive-loss")

    return run_name, save_path, cmd


def main():
    parser = argparse.ArgumentParser(description="GNN merge sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--model-root",    type=Path, default=MODEL_ROOT)
    parser.add_argument("--save-root",     type=Path, default=SAVE_ROOT)
    parser.add_argument("--wandb-project", type=str,  default=WANDB_PROJECT)
    args = parser.parse_args()

    groups = discover_groups(args.model_root)
    if not groups:
        print("[sweep] No valid model groups found — check MODEL_ROOT.", file=sys.stderr)
        sys.exit(1)

    combos  = list(itertools.product(groups, RATIOS, LOSSES))
    total   = len(combos)
    skipped = 0

    print(f"[sweep] Found {len(groups)} merge groups → {total} total runs\n")

    for run_idx, (group, ratio, loss) in enumerate(combos, start=1):
        run_name, save_path, cmd = build_command(
            group, ratio, loss, args.save_root, args.wandb_project
        )

        print(
            f"[{run_idx:>4}/{total}] {run_name}\n"
            f"           save → {save_path}"
        )

        if args.dry_run:
            print(f"           cmd  → {' '.join(cmd)}\n")
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"[sweep] WARNING: run {run_name} exited with code "
                f"{result.returncode} — continuing sweep.",
                file=sys.stderr,
            )
            skipped += 1

    status = "dry-run complete" if args.dry_run else f"done ({skipped} failed)"
    print(f"\n[sweep] {status}. {total} combinations attempted.")


if __name__ == "__main__":
    main()
