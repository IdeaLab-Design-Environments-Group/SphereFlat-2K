"""Load one batch from data/processed via the manifest and print tensor shapes / dtypes / devices.

    uv run python scripts/load_one_batch.py [--manifest PATH] [--processed DIR] [--split train]
        [--balance-by source]

The batch is shown on CPU, then moved with ``to_device``: the grid (float32) goes to mps when available,
float64 per-vertex tensors stay on CPU (MPS has no float64).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from moe_data.config import load_config
from moe_data.dataset import MoebiusSphereDataset, make_loader, to_device

ROOT = Path(__file__).resolve().parents[1]


def describe(batch: dict, label: str) -> None:
    print(f"-- {label}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s} {str(tuple(v.shape)):22s} {str(v.dtype):14s} {v.device}")
        else:
            print(f"  {k:12s} {v}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--manifest", default=None, help="default: paths.manifest from the config")
    ap.add_argument("--processed", default=None, help="default: paths.processed from the config")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-vertices", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--balance-by", choices=["source"], default=None,
                    help="draw batches with equal source-group proportions")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    manifest = Path(args.manifest or ROOT / cfg["paths"]["manifest"])
    processed = Path(args.processed or ROOT / cfg["paths"]["processed"])
    ds = MoebiusSphereDataset(manifest, processed, args.split, cfg, rotate=not args.no_rotate,
                              n_vertices=args.n_vertices, seed=cfg["seed"])
    print(f"{len(ds)} shapes in split '{args.split}' ({manifest})")
    if len(ds) == 0:
        return
    print("groups: " + ", ".join(f"{g}={n}" for g, n in sorted(Counter(ds.groups).items())))
    batch = next(iter(make_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                  seed=cfg["seed"], shuffle=False, balance_by=args.balance_by)))
    print(f"batch source groups: {batch['source_group']}")
    cm = batch["corner_mask"]
    print(f"corner_mask {tuple(cm.shape)}, {int(cm.sum())} corner vertices")
    describe(batch, "cpu")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    describe(to_device(batch, dev), f"to_device('{dev}'), float64 kept on cpu")


if __name__ == "__main__":
    main()
