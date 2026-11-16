"""validate_sphere stage: certify that the SphereMap output P is a valid, centred, single-cover
spherical parameterisation of exactly the mesh that was submitted.

Inputs: V_sub (float32-rounded vertices written to SphereMap, as float64), F, P (sphere positions),
X (x,y,z read back from the SphereMap output), optional F_sphere (faces read back).
Every check is written as accept-if (``ok = all(err <= tol)``) behind an isfinite gate, so NaN fails.
Each failing check contributes its own reason code; all measures are recorded even on failure.
"""
import numpy as np

from moe_data import geometry as geo
from moe_data.stages.base import StageResult

NAME = "validate_sphere"


def _finite_or_nan(x) -> float:
    x = float(x)
    return x if np.isfinite(x) else float("nan")


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    missing = [k for k in ("V_sub", "F", "P", "X") if inputs.get(k) is None]
    if missing:
        return StageResult(False, reason=f"missing_input: {', '.join(missing)}")
    vc = cfg["validate"]
    V_sub = np.asarray(inputs["V_sub"], dtype=np.float64)
    F = np.asarray(inputs["F"], dtype=np.int64)
    P = np.asarray(inputs["P"], dtype=np.float64)
    X = np.asarray(inputs["X"])
    F_sphere = inputs.get("F_sphere")

    if P.shape != V_sub.shape:
        return StageResult(False, meta={"n_P": int(len(P)), "n_V": int(len(V_sub))},
                           reason=f"correspondence_mismatch: P shape {P.shape} != V_sub shape {V_sub.shape}")

    fails: list[str] = []
    meta: dict = {}

    # (1) finite P
    n_nonfinite = int(np.sum(~np.all(np.isfinite(P), axis=1)))
    meta["n_nonfinite_P"] = n_nonfinite
    if not n_nonfinite == 0:
        fails.append(f"nonfinite_P: {n_nonfinite} rows")

    # (2) correspondence: read-back x,y,z bitwise equal to float32(V_sub), same faces
    same_x = X.shape == V_sub.shape and np.array_equal(
        X.astype(np.float32).view(np.uint32), V_sub.astype(np.float32).view(np.uint32))
    same_f = F_sphere is None or np.array_equal(np.asarray(F_sphere, dtype=np.int64), F)
    meta["correspondence_ok"] = bool(same_x and same_f)
    if not (same_x and same_f):
        fails.append(f"correspondence_mismatch: xyz_equal={same_x} faces_equal={same_f}")

    with np.errstate(invalid="ignore", divide="ignore"):
        # (3) unit norm
        norm_err = np.abs(np.linalg.norm(P, axis=1) - 1.0)
        meta["max_norm_err"] = _finite_or_nan(np.max(norm_err))
        if not (np.all(np.isfinite(norm_err)) and np.all(norm_err <= vc["unit_norm_tol"])):
            fails.append(f"off_unit_sphere: max ||p|-1| = {meta['max_norm_err']:.3g}")

        # (4) orientation and non-degeneracy. det[p0,p1,p2] cancels to rounding noise on tiny triangles
        # (false flips from L/r ~ 20), so orientation uses the edge cross product against the
        # triangle centroid, and degeneracy uses the shortest edge (chord) instead of an absolute area.
        p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        orient = geo.robust_spherical_orientation(P, F)
        edge = np.minimum(np.minimum(np.linalg.norm(p1 - p0, axis=1), np.linalg.norm(p2 - p1, axis=1)),
                          np.linalg.norm(p0 - p2, axis=1))
        n_flipped = int(np.sum(~(orient > 0)))
        n_degenerate = int(np.sum(~(edge >= vc["min_sphere_edge"])))
        meta.update(min_orient=_finite_or_nan(np.min(orient)), min_sphere_edge=_finite_or_nan(np.min(edge)),
                    n_flipped=n_flipped, n_degenerate=n_degenerate)
        if not n_flipped == 0:
            fails.append(f"flipped_triangles: {n_flipped} with negative orientation")
        if not n_degenerate == 0:
            fails.append(f"degenerate_sphere_triangles: {n_degenerate} with an edge below min_sphere_edge")

        # (5) coverage: single positive cover has signed area exactly 4*pi (double cover: 8*pi)
        cover_err = float(np.sum(geo.spherical_triangle_areas(P, F)) / (4.0 * np.pi) - 1.0)
        meta["coverage_err"] = _finite_or_nan(cover_err)
        if not (np.isfinite(cover_err) and abs(cover_err) <= cfg["validate"]["coverage_tol"]):
            fails.append(f"coverage_not_single: area/4pi - 1 = {cover_err:.3g}")

        # (6) centering residual on the submitted geometry
        res = float(np.linalg.norm(geo.centering_residual(V_sub, P, F)))
        meta["centering_residual"] = _finite_or_nan(res)
        if not (np.isfinite(res) and res <= vc["centering_tol"]):
            fails.append(f"not_centered: |residual| = {res:.3g}")

    if fails:
        return StageResult(False, meta=meta, reason="; ".join(fails))
    return StageResult(True, data={"P": P}, meta=meta)
