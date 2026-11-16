"""Per-channel statistics of the grids over the train split, and their sensitivity to rotation.

    uv run python scripts/channel_stats.py [--manifest PATH] [--processed DIR] [--out data/channel_stats.json]
        [--n-rot-shapes 200] [--n-rotations 3]

Statistics are over all grid cells of all train shapes (cells weighted equally, as the network sees them,
not by solid angle). For every channel the json holds:

- ``stored``: percentiles p0.1/p1/p50/p99/p99.9, min/max, mean/std, median |x| of the stored grids;
- ``transform_stored``: the recommended transform (``moe_data.dataset.recommend_transform``) with the
  mean / std of the transformed stored grids, used by ``MoebiusSphereDataset(normalize="stats",
  rotate=False)``;
- ``transform``: the same transform with the mean / std of the transformed rotated sample, used with
  rotate=True. The stored frame is not random (Thingi10K shapes keep their modelling axes, and their
  extremes sit at the poles far more often than the polar area share), and the lat-long grid gives
  polar cells a large weight per steradian, so cell statistics of stored and rotated grids differ;
- ``stored_transformed``: the summary of the stored grids after transform and standardisation;
- ``rotation``: for a random sample of train shapes, the summary of the stored grids and of grids
  recomputed at ``--n-rotations`` uniform rotations each (the loader's rotate=True path), the shift of
  each pooled statistic in units of the stored std, and the relative change of each shape's channel
  max / min under rotation (raw and after the recommended transform, denominators floored at 1e-3 std).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moe_data.config import load_config
from moe_data.dataset import (MoebiusSphereDataset, apply_transform, read_processed, recommend_transform,
                              summarize_values)

ROOT = Path(__file__).resolve().parents[1]


def load_stored_grids(ids: list[str], processed: Path) -> np.ndarray:
    """(n, C, H, W) float32 stored grids, reading only the ``grid`` array of each file."""
    out = []
    for sid in ids:
        with np.load(processed / f"{sid}.npz", allow_pickle=False) as z:
            out.append(z["grid"])
    return np.stack(out)


def rel_change(a: np.ndarray, b: np.ndarray, floor: float) -> dict[str, float]:
    """Median / p90 / max over shapes of |b - a| / max(|a|, floor) (a: stored, b: rotated). ``floor``
    keeps extremes at or near zero (e.g. an empty-cell vertex_density of 0) from dividing by zero."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    r = np.abs(b - a) / np.maximum(np.abs(a), floor)
    return {"median": float(np.median(r)), "p90": float(np.percentile(r, 90)), "max": float(r.max())}


def rotation_study(df: pd.DataFrame, processed: Path, cfg: dict, channels: tuple[str, ...],
                   transforms: dict, n_shapes: int, n_rot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(df), size=min(n_shapes, len(df)), replace=False))
    sample = df.iloc[pick]
    ds = MoebiusSphereDataset(sample, processed, "train", cfg, rotate=True, seed=seed)
    stored, rotated = [], []
    for i, sid in enumerate(ds.ids):
        stored.append(read_processed(processed / f"{sid}.npz")["grid"])
        rotated.append(np.stack([ds[i]["grid"].numpy() for _ in range(n_rot)]))
        if (i + 1) % 50 == 0:
            print(f"  rotated {i + 1}/{len(ds)} shapes", flush=True)
    S = np.stack(stored)                       # (n, C, H, W)
    Rg = np.stack(rotated)                     # (n, n_rot, C, H, W)
    out = {}
    for k, c in enumerate(channels):
        t = transforms[c]
        s_raw, r_raw = S[:, k], Rg[:, :, k]
        s_sum, r_sum = summarize_values(s_raw), summarize_values(r_raw)
        shift = {key: (r_sum[key] - s_sum[key]) / s_sum["std"] for key in s_sum if key != "std"}
        shift["std"] = r_sum["std"] / s_sum["std"] - 1.0
        smax, smin = s_raw.max(axis=(1, 2))[:, None], s_raw.min(axis=(1, 2))[:, None]
        rmax, rmin = r_raw.max(axis=(2, 3)), r_raw.min(axis=(2, 3))
        tr = lambda x: apply_transform(x, t["kind"], t["scale"])        # noqa: E731
        floor, tfloor = 1e-3 * s_sum["std"], 1e-3 * t["std"]
        smax, smin = np.broadcast_to(smax, rmax.shape), np.broadcast_to(smin, rmin.shape)
        out[c] = {"sample_stored": s_sum, "sample_rotated": r_sum, "pooled_shift_in_stored_std": shift,
                  "sample_stored_transformed": summarize_values(tr(s_raw)),
                  "sample_rotated_transformed": summarize_values(tr(r_raw)),
                  "shape_max_rel_change": rel_change(smax, rmax, floor),
                  "shape_min_rel_change": rel_change(smin, rmin, floor),
                  "shape_max_rel_change_transformed": rel_change(tr(smax), tr(rmax), tfloor),
                  "shape_min_rel_change_transformed": rel_change(tr(smin), tr(rmin), tfloor)}
    return {"ids": list(ds.ids), "n_rotations": n_rot, "channels": out}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--manifest", default=None, help="default: paths.manifest from the config")
    ap.add_argument("--processed", default=None, help="default: <paths.processed>/full")
    ap.add_argument("--out", default=str(ROOT / "data" / "channel_stats.json"))
    ap.add_argument("--n-rot-shapes", type=int, default=200)
    ap.add_argument("--n-rotations", type=int, default=3)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    manifest = Path(args.manifest or ROOT / cfg["paths"]["manifest"])
    processed = Path(args.processed or ROOT / cfg["paths"]["processed"] / "full")
    df = pd.read_parquet(manifest)
    train = df.loc[df["split"] == "train"].reset_index(drop=True)
    channels = tuple(cfg["grid"]["channels"])
    print(f"{len(train)} train shapes ({manifest}), grids from {processed}")

    G = load_stored_grids(train["id"].astype(str).tolist(), processed)
    if tuple(G.shape[1:]) != (len(channels), cfg["grid"]["height"], cfg["grid"]["width"]):
        raise ValueError(f"stored grids are {G.shape[1:]}, config asks for {len(channels)} channels at "
                         f"{cfg['grid']['height']}x{cfg['grid']['width']}")
    per, transforms = {}, {}
    for k, c in enumerate(channels):
        x = G[:, k].astype(np.float64)
        t = recommend_transform(x)
        y = (apply_transform(x, t["kind"], t["scale"]) - t["mean"]) / t["std"]
        transforms[c] = t
        per[c] = {"stored": summarize_values(x), "transform_stored": t, "stored_transformed": summarize_values(y)}
    del G

    print(f"rotation study: {args.n_rot_shapes} shapes x {args.n_rotations} rotations")
    rot = rotation_study(train, processed, cfg, channels, transforms, args.n_rot_shapes, args.n_rotations,
                         int(cfg["seed"]))
    for c in channels:
        r = per[c]["rotation"] = rot["channels"][c]
        rt = r["sample_rotated_transformed"]
        per[c]["transform"] = {**transforms[c], "mean": rt["mean"], "std": rt["std"] if rt["std"] > 0 else 1.0}
    result = {"meta": {"split": "train", "n_shapes": len(train), "manifest": str(manifest),
                       "processed": str(processed), "height": cfg["grid"]["height"],
                       "width": cfg["grid"]["width"], "cell_weighting": "equal per cell",
                       "rotation_sample_n": len(rot["ids"]), "rotation_n_rotations": rot["n_rotations"],
                       "seed": int(cfg["seed"]),
                       "transform_rule": "asinh(x / median|x|) when max|x| > 100 median|x| over the stored train "
                                         "grids, else identity; then (T(x) - mean) / std",
                       "transform": "mean / std of T(x) over the rotated sample (loader with rotate=True)",
                       "transform_stored": "mean / std of T(x) over all stored train grids (rotate=False)"},
              "channels": per}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}\n")

    hdr = f"{'channel':20s} {'p0.1':>10s} {'p1':>10s} {'p50':>10s} {'p99':>10s} {'p99.9':>10s} " \
          f"{'mean':>10s} {'std':>10s} {'max|x|':>10s}  transform"
    print(hdr)
    for c in channels:
        s, t = per[c]["stored"], per[c]["transform_stored"]
        tdesc = f"asinh(x/{t['scale']:.3g})" if t["kind"] == "asinh" else "identity"
        print(f"{c:20s} " + " ".join(f"{s[k]:10.3g}" for k in ("p0.1", "p1", "p50", "p99", "p99.9", "mean", "std"))
              + f" {max(abs(s['min']), abs(s['max'])):10.3g}  {tdesc}")
    print("\nrotation: relative change of each shape's channel max / min (median, p90), raw -> transformed;"
          "\n          pooled p0.1 / p99.9 / std shift in stored std")
    for c in channels:
        r = per[c]["rotation"]
        mx, mn = r["shape_max_rel_change"], r["shape_min_rel_change"]
        tmx, tmn = r["shape_max_rel_change_transformed"], r["shape_min_rel_change_transformed"]
        sh = r["pooled_shift_in_stored_std"]
        print(f"{c:20s} max {mx['median']:6.1%} {mx['p90']:6.1%} -> {tmx['median']:6.1%} {tmx['p90']:6.1%} | "
              f"min {mn['median']:6.1%} {mn['p90']:6.1%} -> {tmn['median']:6.1%} {tmn['p90']:6.1%} | "
              f"p0.1 {sh['p0.1']:+.3f} p99.9 {sh['p99.9']:+.3f} std {sh['std']:+.1%}")
    print("\ntransformed mean / std: stored train grids -> rotated sample")
    for c in channels:
        a, b = per[c]["transform_stored"], per[c]["transform"]
        print(f"{c:20s} mean {a['mean']:8.3f} -> {b['mean']:8.3f}   std {a['std']:8.3f} -> {b['std']:8.3f}")


if __name__ == "__main__":
    main()
