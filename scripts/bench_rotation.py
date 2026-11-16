"""Benchmark exact per-sample rotation (exact) against grid interpolation (approximate).

Builds 20 synthetic star-shaped meshes (15k vertices: convex hull of jittered Fibonacci points as P,
radial shape V = r(P) P) with the pipeline's own curvature / psi / grid functions, writes them as processed files and
times: one sample in-process, the exact grid alone, torch grid_sample on the stored grid, and DataLoader
throughput for several worker counts.

    uv run python scripts/bench_rotation.py [--out DIR] [--workers 0 4 8 12]
"""
from __future__ import annotations

import argparse
import multiprocessing
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import ConvexHull

from moe_data import geometry as geo
from moe_data import grid as gr
from moe_data.config import load_config
from moe_data.dataset import MoebiusSphereDataset, make_loader, read_processed, write_processed
from moe_data.stages import grid as grid_stage
from moe_data.stages.psi import psi_barycentric

ROOT = Path(__file__).resolve().parents[1]


def sphere_mesh(rng: np.random.Generator, n: int = 15000, jitter: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Irregular unit-sphere mesh: jittered Fibonacci points, randomly rotated, convex hull, faces outward."""
    i = np.arange(n) + 0.5 + rng.uniform(-jitter, jitter, n)
    z = np.clip(1 - 2 * i / n, -1, 1)
    phi = np.pi * (1 + 5**0.5) * np.arange(n) + rng.uniform(-jitter, jitter, n) * 2 / np.sqrt(n)
    P = np.column_stack([np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z])
    P = P @ gr.random_rotation(rng).T
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    F = ConvexHull(P).simplices.astype(np.int64)
    flip = np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])) < 0
    F[flip] = F[flip][:, ::-1]
    return P, F


def synthetic_shape(rng: np.random.Generator, cfg: dict) -> dict:
    P, F = sphere_mesh(rng)
    d = rng.normal(size=(4, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = np.exp(np.square(P @ d.T) @ rng.uniform(-0.4, 0.6, 4))  # smooth radial bumps
    V = P * r[:, None]
    area = geo.vertex_areas_barycentric(V, F)
    K_int = geo.angle_defect(V, F)
    a = {"V": V, "F": F, "P": P, "area": area, "area_sphere": geo.vertex_areas_barycentric(P, F),
         "psi": psi_barycentric(V, F, P), "K_int": K_int, "K_pt": K_int / area, "H": geo.mean_curvature(V, F),
         "vis": rng.random((len(V), cfg["visibility"]["n_view"] + cfg["visibility"]["n_paint"])).astype(np.float32)}
    res = grid_stage.run("bench", a, cfg)
    if not res.ok:
        raise RuntimeError(res.reason)
    return {**a, **res.data}


def build(out: Path, cfg: dict, n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for k in range(n):
        write_processed(out / f"bench_{k:02d}.npz", synthetic_shape(rng, cfg))
        rows.append({"id": f"bench_{k:02d}", "ok": True, "split": "train"})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "manifest.parquet", index=False)
    return df


def _ms(fn, reps: int) -> float:
    t = time.perf_counter()
    for i in range(reps):
        fn(i)
    return (time.perf_counter() - t) / reps * 1e3


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=None, help="keep the synthetic processed files here")
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8, 12])
    ap.add_argument("--samples", type=int, default=120, help="samples per DataLoader run")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    tmp = None
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory()
        out = Path(tmp.name)
    try:
        t0 = time.perf_counter()
        df = build(out, cfg)
        n_v = [len(read_processed(out / f"{i}.npz")["P"]) for i in df["id"]]
        print(f"built {len(df)} shapes, {min(n_v)}-{max(n_v)} vertices, grid "
              f"{cfg['grid']['height']}x{cfg['grid']['width']}x{len(cfg['grid']['channels'])}, "
              f"{time.perf_counter() - t0:.1f}s")

        ds = MoebiusSphereDataset(df, out, "train", cfg, rotate=True, seed=0)
        ds_fixed = MoebiusSphereDataset(df, out, "train", cfg, rotate=False, seed=0)
        rng = np.random.default_rng(1)
        shapes = [read_processed(out / f"{i}.npz") for i in df["id"]]
        gc = cfg["grid"]
        Rs = [gr.random_rotation(rng) for _ in range(20)]

        def exact_grid(i):
            s = shapes[i % 20]
            gr.compute_grid(s["P"], s["F"], s["psi"], s["K_pt"], s["H"], s["K_int"], gc["height"], gc["width"],
                            gc["splat_alpha_cells"], tuple(gc["channels"]), R=Rs[i % 20])

        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        rows = [("item, exact rotation (read + grid + P)", _ms(lambda i: ds[i % 20], 20)),
                ("  of which compute_grid at R", _ms(exact_grid, 20)),
                ("item, no rotation (read only)", _ms(lambda i: ds_fixed[i % 20], 20)),
                ("grid_sample on stored grid, CPU 1 thread", _ms(lambda i: gr.rotate_grid_torch(
                    shapes[i % 20]["grid"], Rs[i % 20]), 20))]
        if torch.backends.mps.is_available():
            G = torch.from_numpy(np.stack([s["grid"] for s in shapes[:8]])).to("mps")

            def mps_batch(i):
                out_ = torch.stack([gr.rotate_grid_torch(G[j].cpu().numpy(), Rs[j], "mps") for j in range(8)])
                torch.mps.synchronize()
                return out_
            mps_batch(0)
            rows.append(("grid_sample on MPS (per sample, incl. host copy)", _ms(mps_batch, 5) / 8))
        torch.set_num_threads(n_threads)
        print("\n| per-sample cost (single process) | ms |\n|---|---|")
        for name, ms in rows:
            print(f"| {name} | {ms:.1f} |")

        reps = int(np.ceil(args.samples / len(df)))
        dfl = pd.concat([df] * reps, ignore_index=True).iloc[:args.samples]
        dsl = MoebiusSphereDataset(dfl, out, "train", cfg, rotate=True, seed=0)
        print(f"\n| num_workers | samples | total s | samples/s (total) | samples/s (after 1st batch) |"
              f" leftover children |\n|---|---|---|---|---|---|")
        for nw in args.workers:
            loader = make_loader(dsl, batch_size=args.batch_size, num_workers=nw, seed=0)
            t = time.perf_counter()
            t1, n = None, 0
            for b in loader:
                n += len(b["id"])
                if t1 is None:
                    t1, n1 = time.perf_counter(), n
            t_end = time.perf_counter()
            del loader, b
            left = len(multiprocessing.active_children())
            print(f"| {nw} | {n} | {t_end - t:.1f} | {n / (t_end - t):.1f} | "
                  f"{(n - n1) / max(t_end - t1, 1e-9):.1f} | {left} |")
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
