"""psi stage: log scale factor of the sphere map, per vertex.

``psi`` is the brief's definition: 0.5*log(A_sphere/A_surface) with barycentric vertex areas (one third
of incident flat triangle areas) on both sides; sphere triangles are chordal. ``psi_lsq`` is the
edge least-squares estimate: for every edge, (psi_i+psi_j)/2 = log(|p_i-p_j|/|v_i-v_j|).
Both are stored; `grid.psi_source` selects the one the grid and loader use. ``lsq_residual_rms`` is the RMS
edge residual of the fit, a per-shape conformality diagnostic (0 for an exactly Möbius map).

``corner_dist`` is each vertex's distance to the nearest sharp input-mesh vertex; the ``*_nocorner_*``
statistics exclude vertices closer than ``corners.exclude_radius`` (psi at corners does not
converge with resolution, so those vertices make psi ranges resolution-dependent).

Protrusion statistics are taken on psi minus its area-weighted mean, because a unit-area round shape
has psi = 0.5*log(4*pi) ~ 1.266, not 0.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import scipy.sparse.linalg as spla

from moe_data import geometry as geo
from moe_data.stages.base import StageResult

NAME = "psi"


def _unique_edges(F: np.ndarray) -> np.ndarray:
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    return np.unique(E, axis=0)


def psi_barycentric(V: np.ndarray, F: np.ndarray, P: np.ndarray) -> np.ndarray:
    return 0.5 * np.log(geo.vertex_areas_barycentric(P, F) / geo.vertex_areas_barycentric(V, F))


def psi_edge_lsq(V: np.ndarray, F: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, float]:
    """(psi, rms edge residual). Normal equations are SPD (the edge graph is not bipartite)."""
    E = _unique_edges(F)
    b = np.log(np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1) / np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1))
    m, n = len(E), len(V)
    A = sp.csr_matrix((np.full(2 * m, 0.5), (np.repeat(np.arange(m), 2), E.ravel())), shape=(m, n))
    psi = spla.spsolve((A.T @ A).tocsc(), A.T @ b)
    return psi, float(np.sqrt(np.mean((A @ psi - b) ** 2)))


def psi_stats(psi: np.ndarray, area: np.ndarray, psi_low: float) -> dict:
    mean = float(np.sum(area * psi) / np.sum(area))
    rel = psi - mean
    return {"mean": mean, "min": float(rel.min()), "max": float(rel.max()),
            "range": float(rel.max() - rel.min()),
            "area_share_low": float(np.sum(area[rel < psi_low]) / np.sum(area))}


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    missing = [k for k in ("V_sub", "F", "P") if inputs.get(k) is None]
    if missing:
        return StageResult(False, reason=f"missing_input: {', '.join(missing)}")
    V = np.asarray(inputs["V_sub"], np.float64)
    F = np.asarray(inputs["F"], np.int64)
    P = np.asarray(inputs["P"], np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = psi_barycentric(V, F, P)
        psi_lsq, resid = psi_edge_lsq(V, F, P)
    if not (np.all(np.isfinite(psi)) and np.all(np.isfinite(psi_lsq)) and np.isfinite(resid)):
        return StageResult(False, meta={"n_nonfinite": int(np.sum(~np.isfinite(psi)))},
                           reason="nonfinite_psi: log of a zero or non-finite area/edge ratio")
    area = geo.vertex_areas_barycentric(V, F)
    low = cfg["protrusion"]["psi_low"]
    meta = {f"psi_{k}": v for k, v in psi_stats(psi, area, low).items()}
    meta.update({f"psi_lsq_{k}": v for k, v in psi_stats(psi_lsq, area, low).items()})
    # distance to the nearest sharp vertex of the input surface (corners stage); psi is log-singular there
    pts = np.asarray(inputs.get("corner_pts", np.zeros((0, 3))), np.float64).reshape(-1, 3)
    corner_dist = cKDTree(pts).query(V)[0] if len(pts) else np.full(len(V), np.inf)
    away = corner_dist >= cfg["corners"]["exclude_radius"]
    meta["corner_area_share"] = float(area[~away].sum() / area.sum())
    if away.any():
        meta.update({f"psi_nocorner_{k}": v for k, v in psi_stats(psi[away], area[away], low).items()})
        meta.update({f"psi_lsq_nocorner_{k}": v for k, v in psi_stats(psi_lsq[away], area[away], low).items()})
        d = (psi - meta["psi_mean"]) - (psi_lsq - meta["psi_lsq_mean"])
        meta["psi_vs_lsq_nocorner_max_abs"] = float(np.max(np.abs(d[away])))
    meta["lsq_residual_rms"] = resid
    meta["psi_vs_lsq_max_abs"] = float(np.max(np.abs((psi - meta["psi_mean"]) - (psi_lsq - meta["psi_lsq_mean"]))))
    return StageResult(True, data={"psi": psi, "psi_lsq": psi_lsq, "corner_dist": corner_dist}, meta=meta)
