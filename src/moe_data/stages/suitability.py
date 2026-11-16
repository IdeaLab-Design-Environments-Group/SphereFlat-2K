"""Suitability stage: reject shapes too coarse, too flat/needle-like or too non-compact.

Measures (always recorded in meta, also on rejection, so the pilot can histogram them):
  n_faces, area, volume, compactness = volume / area**1.5 (round sphere ~0.0940),
  pca_ratio_min = smallest / largest and pca_ratio_mid = middle / largest of the
  square roots of the eigenvalues of the area-weighted surface covariance.
"""
import numpy as np

from moe_data.stages.base import StageResult

NAME = "suitability"


def surface_stats(V, F):
    """Exact area-weighted surface moments of a triangle mesh.

    Returns dict with ``area`` (total), ``volume`` (signed, divergence theorem),
    ``centroid`` (3,) area-weighted surface centroid and ``extents`` (3,) the square
    roots of the eigenvalues of the surface covariance, sorted ascending.
    Uses the exact triangle second moment  A/12 * (sum v_i v_i^T + s s^T), s = sum v_i.
    """
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(cr, axis=1)
    area = float(areas.sum())
    volume = float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)
    if not area > 0:
        return {"area": area, "volume": volume, "centroid": np.full(3, np.nan),
                "extents": np.full(3, np.nan)}
    s = a + b + c
    centroid = (areas[:, None] * s).sum(0) / (3.0 * area)
    second = (np.einsum("t,ti,tj->ij", areas, a, a) + np.einsum("t,ti,tj->ij", areas, b, b)
              + np.einsum("t,ti,tj->ij", areas, c, c) + np.einsum("t,ti,tj->ij", areas, s, s)) / 12.0
    cov = second / area - np.outer(centroid, centroid)
    ev = np.clip(np.linalg.eigvalsh(0.5 * (cov + cov.T)), 0.0, None)
    return {"area": area, "volume": volume, "centroid": centroid, "extents": np.sqrt(ev)}


def measures(V, F) -> dict:
    st = surface_stats(V, F)
    area, vol, ext = st["area"], st["volume"], st["extents"]
    largest = ext[2]
    ok_area = area > 0 and np.isfinite(area)
    return {
        "n_faces": int(F.shape[0]),
        "area": float(area),
        "volume": float(vol),
        "compactness": float(abs(vol) / area**1.5) if ok_area else float("nan"),
        "pca_ratio_min": float(ext[0] / largest) if ok_area and largest > 0 else float("nan"),
        "pca_ratio_mid": float(ext[1] / largest) if ok_area and largest > 0 else float("nan"),
    }


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    V, F = inputs.get("V"), inputs.get("F")
    if V is None or F is None:
        return StageResult(False, reason="missing_input: V or F not in inputs")
    th = cfg["suitability"]
    meta = measures(np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64))
    if not (meta["area"] > 0 and np.isfinite(meta["compactness"])):
        return StageResult(False, meta=meta, reason=f"zero_area: area={meta['area']:.3g}")
    if meta["n_faces"] < th["min_faces"]:
        return StageResult(False, meta=meta,
                           reason=f"too_few_faces: {meta['n_faces']} < {th['min_faces']}")
    if meta["compactness"] < th["min_compactness"]:
        return StageResult(False, meta=meta, reason=(
            f"not_compact: compactness={meta['compactness']:.4g} < {th['min_compactness']}"))
    if not meta["pca_ratio_min"] >= th["min_pca_ratio"]:
        return StageResult(False, meta=meta, reason=(
            f"too_thin: pca_ratio_min={meta['pca_ratio_min']:.4g} < {th['min_pca_ratio']}"))
    return StageResult(True, meta=meta)
