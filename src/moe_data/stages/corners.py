"""corners stage: sharp vertices of the (normalised, pre-remesh) input surface.

At a vertex with angle defect K the conformal factor is log-singular, so psi there drifts with mesh size
by K/(2*pi - K) per unit log h and never converges. This stage records the input
mesh's corner points (|angle defect| > corners.k_threshold); the psi stage turns them into a per-vertex
distance ``corner_dist`` on the remeshed surface so the loss can mask or down-weight corner vertices.
Runs before remesh, which rescales the mesh to unit area by a factor within ~0.1% of 1 (pilot remesh area
ratio >= 0.987), so the corner positions are accurate to well below the exclusion radius.
"""
import numpy as np

from moe_data import geometry as geo
from moe_data.stages.base import StageResult

NAME = "corners"


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    if inputs.get("V") is None or inputs.get("F") is None:
        return StageResult(False, reason="missing_input: V or F not in inputs")
    V = np.asarray(inputs["V"], np.float64)
    F = np.asarray(inputs["F"], np.int64)
    K = geo.angle_defect(V, F)
    if not np.all(np.isfinite(K)):
        return StageResult(False, reason="nonfinite_angle_defect: input mesh")
    sharp = np.abs(K) > cfg["corners"]["k_threshold"]
    pts = V[sharp]
    meta = {"n_corners": int(sharp.sum()), "max_abs_defect": float(np.abs(K).max())}
    return StageResult(True, data={"corner_pts": pts.reshape(-1, 3)}, meta=meta)
