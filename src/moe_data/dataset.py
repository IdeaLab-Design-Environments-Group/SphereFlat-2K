"""PyTorch dataset over ``data/processed/<id>.npz`` (format in docs/CONTRACTS.md).

Every sample is in one frame: with ``rotate=True`` a uniform proper rotation R is drawn, P is rotated
exactly (P @ R.T) and the grid is recomputed exactly from the mesh at R (the stored grid
is not interpolated). ``R`` is returned so a consumer can map back.

Precision: P, area and psi stay float64 on CPU. MPS has no float64 (moving a float64 tensor to mps
raises TypeError), so the training loop must either compute the per-vertex loss on CPU or cast only
*relative* quantities (differences, centred values) to float32 before moving them. ``to_device`` does
the safe default: float32 tensors go to the device, float64 tensors stay on CPU unless
``float64="cast"`` is passed.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from moe_data import grid as gr

REQUIRED_KEYS = ("V", "F", "P", "area", "area_sphere", "psi", "K_int", "K_pt", "H", "vis", "grid",
                 "grid_channels")
_DTYPES = {"V": np.float64, "F": np.int64, "P": np.float64, "area": np.float64, "area_sphere": np.float64,
           "psi": np.float64, "K_int": np.float64, "K_pt": np.float64, "H": np.float64, "vis": np.float32,
           "grid": np.float32}
PER_VERTEX = ("P", "area", "psi", "vis", "index", "corner_mask")
SOURCE_GROUPS = ("thingi10k", "proc_blob", "proc_protrusion")


def source_groups(df: pd.DataFrame) -> list[str]:
    """Per-row group: ``source`` for scanned shapes, ``proc_<load.family>`` for procedural ones.

    Rows without a ``source`` column give ``"unknown"``.
    """
    if "source" not in df.columns:
        return ["unknown"] * len(df)
    fam = df["load.family"] if "load.family" in df.columns else pd.Series([None] * len(df), index=df.index)
    out = []
    for src, f in zip(df["source"], fam):
        src = str(src)
        out.append(f"proc_{f}" if src == "procedural" and isinstance(f, str) and f else src)
    return out


def write_processed(path: str | Path, arrays: dict) -> None:
    """Write one shape's final npz (compressed, atomic). Extra keys (e.g. psi_lsq) are kept."""
    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    if missing:
        raise KeyError(f"processed file needs {missing}")
    out = {k: np.asarray(v) for k, v in arrays.items()}
    for k, dt in _DTYPES.items():
        out[k] = out[k].astype(dt, copy=False)
    out["grid_channels"] = np.asarray(out["grid_channels"], dtype=str)
    n = len(out["P"])
    if out["vis"].ndim != 2 or out["vis"].shape[0] != n:
        raise ValueError(f"vis must be (n, n_vis) with n = {n}, got {out['vis'].shape}")
    if out["grid"].shape[0] != len(out["grid_channels"]):
        raise ValueError("grid channel count does not match grid_channels")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)


def read_processed(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    return d


class MoebiusSphereDataset(Dataset):
    """One item per shape of ``split``. Items are dicts:

    grid (C,H,W) float32, P (n,3) float64, area (n,) float64 quadrature weights (barycentric vertex areas,
    or total_area / n_vertices after subsampling, so sum(area * f) estimates the surface integral of f),
    psi (n,) float64 (the field named by ``grid.psi_source``), vis (n, n_vis) float32, index (n,) int64
    (vertex ids in the file), corner_mask (n,) bool (True where the vertex lies closer than the corner
    radius to a sharp input-mesh vertex; all False when the file has no ``corner_dist``), R (3,3) float64,
    id str, source_group str (see ``source_groups``).

    ``corner_radius`` overrides ``cfg['corners']['exclude_radius']``.
    """

    def __init__(self, manifest_path_or_df, processed_dir, split: str, cfg: dict, rotate: bool = True,
                 n_vertices: int | None = None, seed: int = 0, corner_radius: float | None = None):
        df = manifest_path_or_df
        if not isinstance(df, pd.DataFrame):
            df = pd.read_parquet(df)
        if "split" not in df.columns:
            raise KeyError("manifest has no 'split' column")
        rows = df.loc[df["split"] == split]
        self.ids = rows["id"].astype(str).tolist()
        self.groups = source_groups(rows)
        self.processed_dir = Path(processed_dir)
        gc = cfg["grid"]
        self.height, self.width = int(gc["height"]), int(gc["width"])
        self.alpha_cells = float(gc["splat_alpha_cells"])
        self.channels = tuple(gc["channels"])
        self.psi_key = gc["psi_source"]
        self.rotate, self.n_vertices, self.seed = rotate, n_vertices, seed
        self.rng = np.random.default_rng(seed)
        self.corner_radius = float(cfg["corners"]["exclude_radius"] if corner_radius is None else corner_radius)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        sid = self.ids[i]
        d = read_processed(self.processed_dir / f"{sid}.npz")
        P, F = d["P"], d["F"]
        if self.psi_key not in d:
            raise KeyError(f"{sid}: grid.psi_source '{self.psi_key}' not in processed file")
        psi = d[self.psi_key]
        stored = str(d["grid_psi_source"]) if "grid_psi_source" in d else "psi"
        if not self.rotate and stored != self.psi_key:
            raise ValueError(f"{sid}: stored grid was built from '{stored}' but grid.psi_source is "
                             f"'{self.psi_key}'; rebuild the grids or use rotate=True")
        if self.rotate:
            R = gr.random_rotation(self.rng)
            assert np.linalg.det(R) > 0 and np.allclose(R @ R.T, np.eye(3), atol=1e-12)
            G, info = gr.compute_grid(P, F, psi, d["K_pt"], d["H"], d["K_int"], self.height, self.width,
                                      self.alpha_cells, self.channels, R=R)
            if info["n_unlocated"]:
                raise RuntimeError(f"{sid}: {info['n_unlocated']} cell centres not located")
            P = P @ R.T
        else:
            R = np.eye(3)
            stored = tuple(str(c) for c in d["grid_channels"])
            if stored != self.channels:
                raise ValueError(f"{sid}: stored grid channels {stored} != config {self.channels}")
            G = d["grid"]
        area = d["area"]
        corner = (d["corner_dist"] < self.corner_radius if "corner_dist" in d
                  else np.zeros(len(P), dtype=bool))
        idx = np.arange(len(P))
        if self.n_vertices is not None:
            idx, area = area_weighted_subsample(area, self.n_vertices, self.rng)
        return {"id": sid,
                "grid": torch.from_numpy(np.ascontiguousarray(G, dtype=np.float32)),
                "P": torch.from_numpy(np.ascontiguousarray(P[idx], dtype=np.float64)),
                "area": torch.from_numpy(np.ascontiguousarray(area, dtype=np.float64)),
                "psi": torch.from_numpy(np.ascontiguousarray(psi[idx], dtype=np.float64)),
                "vis": torch.from_numpy(np.ascontiguousarray(d["vis"][idx], dtype=np.float32)),
                "index": torch.from_numpy(idx.astype(np.int64)),
                "corner_mask": torch.from_numpy(np.ascontiguousarray(corner[idx], dtype=bool)),
                "R": torch.from_numpy(R.astype(np.float64)),
                "source_group": self.groups[i]}


def area_weighted_subsample(area: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n vertex ids drawn with replacement, P(v) = area_v / total; weights total / n each.

    E[sum_k w f(v_k)] = n * (total/n) * sum_v (area_v/total) f_v = sum_v area_v f_v (unbiased).
    """
    area = np.asarray(area, np.float64)
    total = area.sum()
    idx = rng.choice(len(area), size=n, replace=True, p=area / total)
    return idx, np.full(n, total / n)


def collate(batch: list[dict]) -> dict:
    """Stack grids and R; pad per-vertex tensors to the longest item with zeros (False for corner_mask);
    ``mask`` (B, n_max) bool marks valid entries."""
    n = [len(b["P"]) for b in batch]
    nmax = max(n)
    out = {"id": [b["id"] for b in batch],
           "source_group": [b["source_group"] for b in batch],
           "grid": torch.stack([b["grid"] for b in batch]),
           "R": torch.stack([b["R"] for b in batch]),
           "mask": torch.arange(nmax)[None, :] < torch.tensor(n)[:, None]}
    for k in PER_VERTEX:
        x0 = batch[0][k]
        pad = x0.new_zeros((len(batch), nmax) + tuple(x0.shape[1:]))
        for j, b in enumerate(batch):
            pad[j, :n[j]] = b[k]
        out[k] = pad
    return out


class SourceBalancedSampler(Sampler[list[int]]):
    """Batch sampler whose batches hold the source groups (``dataset.groups``) in fixed proportions.

    ``weights`` (default equal) is a mapping group -> weight or a sequence aligned with ``group_names``
    (``SOURCE_GROUPS`` order for the groups present, then any other groups sorted). Each batch gets
    floor(b * w_g) items of group g, and the remaining slots go to groups drawn without replacement with
    probability proportional to the fractional parts, so the expected count is exactly b * w_g. Items
    within a group are drawn uniformly with replacement. An epoch has ceil(len(dataset) / batch_size)
    batches and len(dataset) items in total; the draw depends only on (seed, epoch) (``set_epoch``).
    Batches are built in the main process, so it works unchanged with DataLoader workers.
    """

    def __init__(self, dataset, batch_size: int, weights: Mapping[str, float] | Sequence[float] | None = None,
                 seed: int = 0):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        groups = list(dataset.groups)
        if not groups:
            raise ValueError("dataset is empty")
        present = set(groups)
        self.group_names = [g for g in SOURCE_GROUPS if g in present] + sorted(present - set(SOURCE_GROUPS))
        self.members = [np.flatnonzero(np.array(groups) == g) for g in self.group_names]
        if weights is None:
            w = np.ones(len(self.group_names))
        elif isinstance(weights, Mapping):
            unknown = set(weights) - present
            if unknown:
                raise KeyError(f"weights name groups not in the dataset: {sorted(unknown)}")
            w = np.array([float(weights.get(g, 0.0)) for g in self.group_names])
        else:
            w = np.asarray(weights, np.float64)
            if w.shape != (len(self.group_names),):
                raise ValueError(f"need {len(self.group_names)} weights for groups {self.group_names}")
        if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("weights must be finite, >= 0 and not all zero")
        self.weights = w / w.sum()
        self.batch_size, self.seed, self.epoch = int(batch_size), int(seed), 0
        self.n_items = len(groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(self.n_items / self.batch_size)

    def _counts(self, b: int, rng: np.random.Generator) -> np.ndarray:
        exact = b * self.weights
        counts = np.floor(exact).astype(int)
        rest = b - counts.sum()
        if rest:
            frac = exact - counts
            extra = rng.choice(len(counts), size=rest, replace=False, p=frac / frac.sum())
            counts[extra] += 1
        return counts

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        for k in range(len(self)):
            b = min(self.batch_size, self.n_items - k * self.batch_size)
            batch = np.concatenate([rng.choice(m, size=c, replace=True)
                                    for m, c in zip(self.members, self._counts(b, rng)) if c])
            yield rng.permutation(batch).tolist()


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own rotation stream, seeded from (dataset seed, torch worker seed).

    The torch worker seed is base_seed + worker_id with base_seed drawn from the loader's generator, so
    workers differ, and a loader built with a seeded generator (``make_loader``) is reproducible.
    """
    info = get_worker_info()
    ds = info.dataset
    ds.rng = np.random.default_rng([ds.seed, info.seed])


def _collate_numpy(batch: list[dict]) -> dict:
    """Worker-side collate: same as ``collate`` but returns NumPy arrays (see ``make_loader``)."""
    return {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in collate(batch).items()}


class SphereLoader:
    """Iterable over collated torch batches; wraps the DataLoader in ``.loader``.

    With workers, batches cross the process boundary as NumPy arrays and become tensors here. On macOS
    with torch 2.15 (sharing strategy file_system, the only one available) a worker that has sent torch
    tensors does not exit at shutdown: the DataLoader waits 5 s per worker, sequentially, then kills it.
    Workers that send NumPy arrays exit at once.
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def batch_sampler(self):
        return self.loader.batch_sampler

    def __iter__(self):
        for b in self.loader:
            yield {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in b.items()}


def make_loader(ds: MoebiusSphereDataset, batch_size: int, num_workers: int = 0, seed: int = 0,
                shuffle: bool = True, balance_by: str | None = None,
                balance_weights: Mapping[str, float] | Sequence[float] | None = None, **kw) -> SphereLoader:
    """Batches via ``collate``, per-worker rotation seeding and a seeded generator (reproducible for a
    fixed seed and num_workers). Iterating to the end shuts the workers down (unless persistent_workers).

    ``balance_by="source"`` draws batches with ``SourceBalancedSampler(ds, batch_size, balance_weights,
    seed)`` (``shuffle`` is then ignored; call ``loader.batch_sampler.set_epoch`` each epoch).
    """
    common = dict(num_workers=num_workers, collate_fn=_collate_numpy if num_workers > 0 else collate,
                  worker_init_fn=worker_init_fn if num_workers > 0 else None,
                  generator=torch.Generator().manual_seed(seed), **kw)
    if balance_by is None:
        if balance_weights is not None:
            raise ValueError("balance_weights needs balance_by")
        return SphereLoader(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **common))
    if balance_by != "source":
        raise ValueError(f"balance_by must be None or 'source', got {balance_by!r}")
    sampler = SourceBalancedSampler(ds, batch_size, balance_weights, seed)
    return SphereLoader(DataLoader(ds, batch_sampler=sampler, **common))


def to_device(batch: dict, device: str | torch.device, float64: str = "cpu") -> dict:
    """Move a collated batch. float64 tensors stay on CPU (``float64="cpu"``) or are cast to float32
    first (``"cast"``; only safe for relative quantities, since float32 has ~7 digits)."""
    if float64 not in ("cpu", "cast"):
        raise ValueError("float64 must be 'cpu' or 'cast'")
    out = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
        elif v.dtype == torch.float64:
            out[k] = v.to(device=device, dtype=torch.float32) if float64 == "cast" else v
        else:
            out[k] = v.to(device)
    return out
