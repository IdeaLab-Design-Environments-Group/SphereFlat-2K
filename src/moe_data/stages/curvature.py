"""curvature stage: integrated Gaussian curvature (angle defect), pointwise K and mean curvature H.

Computed on V_sub (the geometry submitted to SphereMap, so it corresponds to P) when present,
else on V. Gauss-Bonnet (sum K_int = 4*pi for genus 0) is checked in accept-if form, and every
output passes an isfinite gate.
"""
import numpy as np

from moe_data import geometry as geo
from moe_data.stages.base import StageResult

NAME = "curvature"
_QUANTILES = (0.0, 0.01, 0.5, 0.99, 1.0)


def _quantiles(x, name: str) -> dict:
    q = np.quantile(x, _QUANTILES) if np.all(np.isfinite(x)) else np.full(len(_QUANTILES), np.nan)
    return {f"{name}_q{int(round(p * 100)):02d}": float(v) for p, v in zip(_QUANTILES, q)}


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    V = inputs.get("V_sub", inputs.get("V"))
    F = inputs.get("F")
    if V is None or F is None:
        return StageResult(False, reason="missing_input: V_sub/V or F not in inputs")
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    with np.errstate(invalid="ignore", divide="ignore"):
        area = geo.vertex_areas_barycentric(V, F)
        K_int = geo.angle_defect(V, F)
        K_pt = K_int / area
        H = geo.mean_curvature(V, F)

    gb_err = float(abs(np.sum(K_int) - 4.0 * np.pi))
    meta = {"gauss_bonnet_err": gb_err if np.isfinite(gb_err) else float("nan")}
    meta.update(_quantiles(K_pt, "K_pt"))
    meta.update(_quantiles(H, "H"))

    fails = []
    for name, arr in (("area", area), ("K_int", K_int), ("K_pt", K_pt), ("H", H)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        if not n_bad == 0:
            fails.append(f"nonfinite_{name}: {n_bad} vertices")
    if not (np.isfinite(gb_err) and gb_err <= cfg["curvature"]["gauss_bonnet_tol"]):
        fails.append(f"gauss_bonnet: |sum K_int - 4pi| = {gb_err:.3g}")
    if fails:
        return StageResult(False, meta=meta, reason="; ".join(fails))
    return StageResult(True, data={"area": area, "K_int": K_int, "K_pt": K_pt, "H": H}, meta=meta)
