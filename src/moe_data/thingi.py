"""Thingi10K candidate listing and mesh loading (npz variant).

Metadata is only used for bookkeeping (licence, author, thing_id) and for reporting
agreement with our own topology check; it never filters candidates.
"""
from pathlib import Path

import numpy as np
import thingi10k

# metadata column -> candidate key (prefixed: these are the dataset's claims, not ours)
_META_TOPOLOGY = ("num_vertices", "num_facets", "num_components", "num_boundary_edges",
                  "closed", "vertex_manifold", "edge_manifold", "oriented", "PWN", "solid",
                  "euler")
_COLUMNS = ("file_id", "thing_id", "file_path", "author", "license", "self_intersecting",
            *_META_TOPOLOGY)


def is_redistributable(licence) -> bool:
    """False for any non-commercial or no-derivatives licence and for unknown licences."""
    if not isinstance(licence, str) or not licence.strip():
        return False
    s = licence.lower()
    return not ("non-commercial" in s or "no derivatives" in s or "unknown" in s)


def _dataset(cfg):
    thingi10k.init(variant="npz", cache_dir=str(cfg["paths"]["thingi10k_cache"]))
    return thingi10k.dataset()


def candidate_from_row(row: dict) -> dict:
    fid = int(row["file_id"])
    cand = {
        "shape_id": f"thingi_{fid}",
        "file_id": fid,
        "thing_id": int(row["thing_id"]),
        "licence": row["license"],
        "redistributable": is_redistributable(row["license"]),
        "author": row["author"],
        "file_path": str(Path(row["file_path"]).resolve()),
        "self_intersecting": bool(row["self_intersecting"]),
    }
    for k in _META_TOPOLOGY:
        v = row[k]
        cand[f"meta_{k}"] = bool(v) if isinstance(v, (bool, np.bool_)) else int(v)
    return cand


def list_candidates(cfg: dict, pilot: bool) -> list[dict]:
    """One dict per Thingi10K file, sorted by file_id.

    With ``pilot`` a seeded (cfg['seed']) random sample of cfg['run']['pilot_thingi']
    files drawn from all files, without any prefiltering.
    """
    d = _dataset(cfg).select_columns(list(_COLUMNS))
    rows = sorted(d.to_list(), key=lambda r: int(r["file_id"]))
    if pilot:
        k = min(int(cfg["run"]["pilot_thingi"]), len(rows))
        rng = np.random.default_rng(cfg["seed"])
        idx = np.sort(rng.choice(len(rows), size=k, replace=False))
        rows = [rows[i] for i in idx]
    return [candidate_from_row(r) for r in rows]


def load_mesh(candidate: dict):
    """(V float64 (n,3), F int64 (m,3)). Raises FileNotFoundError / ValueError on bad files."""
    V, F = thingi10k.load_file(candidate["file_path"])
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64)


def metadata_topology_ok(row: dict) -> tuple[bool, str]:
    """What the dataset metadata claims about topology, using our reason codes.

    Accepts a raw dataset row or a candidate dict (``meta_`` prefixed keys). Only for
    reporting agreement with :func:`moe_data.topology.check_topology`.
    """
    def g(k):
        return row[f"meta_{k}"] if f"meta_{k}" in row else row[k]

    if not g("closed") or g("num_boundary_edges") > 0:
        return False, f"not_closed: {g('num_boundary_edges')} boundary edges (metadata)"
    if not g("edge_manifold"):
        return False, "edge_nonmanifold: metadata edge_manifold=False"
    if not g("vertex_manifold"):
        return False, "vertex_nonmanifold: metadata vertex_manifold=False"
    if not g("oriented"):
        return False, "inconsistent_orientation: metadata oriented=False"
    if g("num_components") != 1:
        return False, f"multiple_components: {g('num_components')} (metadata)"
    if g("euler") != 2:
        return False, f"not_genus0: chi={g('euler')} (metadata)"
    return True, ""
