"""Normalize stage: area-weighted surface centroid to the origin, total area to 1."""
import numpy as np

from moe_data.stages.base import StageResult
from moe_data.stages.suitability import surface_stats

NAME = "normalize"


def normalize_mesh(V, F):
    """Return (V_normalised, original_area, scale, original_centroid)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    st = surface_stats(V, F)
    area = st["area"]
    if not (area > 0 and np.isfinite(area)):
        return None, area, float("nan"), st["centroid"]
    scale = 1.0 / np.sqrt(area)
    Vn = (V - st["centroid"]) * scale
    # second pass removes the O(eps) residual of the first so the centroid is ~1e-17
    Vn -= surface_stats(Vn, F)["centroid"]
    return Vn, area, scale, st["centroid"]


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    V, F = inputs.get("V"), inputs.get("F")
    if V is None or F is None:
        return StageResult(False, reason="missing_input: V or F not in inputs")
    Vn, area, scale, c = normalize_mesh(V, F)
    if Vn is None:
        return StageResult(False, meta={"orig_area": float(area)},
                           reason=f"zero_area: area={area:.3g}")
    meta = {"orig_area": float(area), "scale": float(scale),
            "orig_centroid_x": float(c[0]), "orig_centroid_y": float(c[1]),
            "orig_centroid_z": float(c[2])}
    return StageResult(True, data={"V": Vn, "F": np.asarray(F, dtype=np.int64)}, meta=meta)
