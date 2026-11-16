"""Load one batch from data/processed/full via the manifest and print tensor shapes / dtypes / devices.

    uv run python scripts/load_one_batch.py [--manifest PATH] [--processed DIR] [--split train]
        [--balance-by source] [--normalize stats] [--return-surface]

The batch is shown on CPU, then moved with ``to_device``: the grid (float32) goes to mps when available,
every per-vertex tensor stays on CPU (MPS has no float64, and the per-vertex loss runs on CPU).
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
    ap.add_argument("--processed", default=None, help="default: <paths.processed>/full, where the full run writes")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-vertices", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--balance-by", choices=["source"], default=None,
                    help="draw batches with equal source-group proportions")
    ap.add_argument("--normalize", choices=["stats"], default=None,
                    help="per-channel transform and standardisation from --channel-stats")
    ap.add_argument("--channel-stats", default=str(ROOT / "data" / "channel_stats.json"))
    ap.add_argument("--return-surface", action="store_true", help="also return surface positions V")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    manifest = Path(args.manifest or ROOT / cfg["paths"]["manifest"])
    processed = Path(args.processed or ROOT / cfg["paths"]["processed"] / "full")
    ds = MoebiusSphereDataset(manifest, processed, args.split, cfg, rotate=not args.no_rotate,
                              n_vertices=args.n_vertices, seed=cfg["seed"], return_surface=args.return_surface,
                              normalize=args.normalize,
                              channel_stats=args.channel_stats if args.normalize else None)
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
    g = batch["grid"]
    print("grid per channel (mean, std, min, max): " + "; ".join(
        f"{c} {float(g[:, k].mean()):.3g} {float(g[:, k].std()):.3g} {float(g[:, k].min()):.3g} "
        f"{float(g[:, k].max()):.3g}" for k, c in enumerate(ds.channels)))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    describe(to_device(batch, dev), f"to_device('{dev}'), per-vertex tensors kept on cpu")


if __name__ == "__main__":
    main()
