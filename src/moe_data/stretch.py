"""Consistency check of the cone-layout log scale factor u (RESEARCH.md formula) against a mesh Poisson solve.

For cones at sphere points p_i with curvatures theta_i (sum 4*pi), the flat-with-cones metric e^{2u} g_M has

    u(x) = psi(x) - (1/(2*pi)) * sum_i theta_i * log||x - p_i|| + c          (closed form, x on the unit sphere)

Derivation (Delta = Laplace-Beltrami, negative semi-definite):
  g_S = e^{2 psi} g_M and K_S = 1  =>  Delta_M psi = K_M - e^{2 psi};
  on the unit sphere Delta_S log||x - p|| = 2*pi*delta_p - 1/2;  Delta_M = e^{2 psi} Delta_S, delta_M = e^{2 psi} delta_S;
  with sum theta = 4*pi the e^{2 psi} terms cancel:  Delta_M u = K_M - sum_i theta_i delta_{p_i}.
Discretely, with the cotan Laplacian L ((L u)_i ~ integral of Delta u over A_i) and the angle defect K_int:

    L u = K_int - Theta,   Theta_i = theta at the cone vertices,                 (Poisson form)

so the two forms must agree up to a constant away from the cones (where both are log-singular) and away from
input-mesh corners (where psi is not resolution-converged).
"""
import numpy as np
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from moe_data import geometry as geo


def closed_form_u(P, psi, cone_idx, theta) -> np.ndarray:
    """psi - (1/2pi) sum theta_i log||P - P[cone_i]||, without the free constant. NaN at the cone vertices."""
    P = np.asarray(P, np.float64)
    cone_idx = np.asarray(cone_idx, np.int64)
    theta = np.asarray(theta, np.float64)
    u = np.asarray(psi, np.float64).copy()
    with np.errstate(divide="ignore"):
        for i, t in zip(cone_idx, theta):
            u -= t / (2.0 * np.pi) * np.log(np.linalg.norm(P - P[i], axis=1))
    u[cone_idx] = np.nan
    return u


def poisson_u(V, F, cone_idx, theta, K_int=None) -> np.ndarray:
    """Solve L u = K_int - Theta on the mesh (L = geometry.cotan_laplacian, negative semi-definite).

    The right-hand side must sum to zero (Gauss-Bonnet with sum theta = 4*pi); any rounding residual is
    removed in proportion to the barycentric vertex area. The constant nullspace is removed by pinning
    vertex 0 and the result is shifted to zero area-weighted mean.
    """
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    K = geo.angle_defect(V, F) if K_int is None else np.asarray(K_int, np.float64)
    area = geo.vertex_areas_barycentric(V, F)
    b = K.copy()
    np.add.at(b, np.asarray(cone_idx, np.int64), -np.asarray(theta, np.float64))
    b -= area * b.sum() / area.sum()
    A = (-geo.cotan_laplacian(V, F)).tocsc()[1:, 1:]       # SPD after pinning u_0 = 0
    u = np.zeros(len(V))
    u[1:] = spla.spsolve(A, -b[1:])
    return u - np.sum(area * u) / area.sum()


def farthest_point_cones(P, k: int, start: int = 0) -> np.ndarray:
    """k farthest-point samples of the sphere positions P, starting at vertex ``start``."""
    P = np.asarray(P, np.float64)
    idx = [int(start)]
    d = np.linalg.norm(P - P[start], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(P - P[idx[-1]], axis=1))
    return np.asarray(idx, np.int64)


def compressed_cones(V, psi, k: int, min_sep: float, candidates=None) -> np.ndarray:
    """k vertices of lowest psi, greedily, each at 3D surface distance >= min_sep from those already chosen.

    ``candidates`` (bool mask) restricts the choice, e.g. to vertices outside the corner mask.
    """
    V = np.asarray(V, np.float64)
    order = np.argsort(np.asarray(psi, np.float64), kind="stable")
    if candidates is not None:
        order = order[np.asarray(candidates, bool)[order]]
    idx: list[int] = []
    for i in order:
        if all(np.linalg.norm(V[i] - V[j]) >= min_sep for j in idx):
            idx.append(int(i))
            if len(idx) == k:
                break
    if len(idx) < k:
        raise ValueError(f"only {len(idx)} of {k} cones with separation {min_sep}")
    return np.asarray(idx, np.int64)


def _stats(d, w, mask) -> dict:
    d, w = d[mask], w[mask]
    d = d - np.sum(w * d) / w.sum()                        # remove the free constant
    a = np.abs(d)
    return {"max": float(a.max()), "p99": float(np.percentile(a, 99)),
            "rms": float(np.sqrt(np.sum(w * d * d) / w.sum())), "n": int(mask.sum()),
            "area_share": float(w.sum())}


def compare(V, F, P, psi, corner_dist, cone_idx, theta, exclude_radius, K_int=None) -> dict:
    """|u_closed - u_poisson| statistics (after removing the area-weighted mean difference).

    ``masked``: vertices with corner_dist >= exclude_radius and 3D distance >= exclude_radius from every cone.
    ``unmasked``: only the cone disks excluded (corner mask NOT applied).
    ``u_range`` / ``u_dev_p99``: range and p99 of |u - area-weighted mean| of the closed form on the masked set,
    for scale. Area shares are fractions of the total barycentric area.
    """
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    cone_idx = np.asarray(cone_idx, np.int64)
    area = geo.vertex_areas_barycentric(V, F)
    w = area / area.sum()
    uc = closed_form_u(P, psi, cone_idx, theta)
    up = poisson_u(V, F, cone_idx, theta, K_int)
    cone_d = cKDTree(V[cone_idx]).query(V)[0]
    near_cone = (cone_d < exclude_radius) | ~np.isfinite(uc)
    away_corner = np.asarray(corner_dist, np.float64) >= exclude_radius
    masked = ~near_cone & away_corner
    unmasked = ~near_cone
    if not masked.any():
        raise ValueError("no vertex left after the corner and cone masks")
    d = uc - up
    ucm = uc[masked] - np.sum(w[masked] * uc[masked]) / w[masked].sum()
    out = {f"masked_{k}": v for k, v in _stats(d, w, masked).items()}
    out.update({f"unmasked_{k}": v for k, v in _stats(d, w, unmasked).items()})
    out["u_range"] = float(ucm.max() - ucm.min())
    out["u_dev_p99"] = float(np.percentile(np.abs(ucm), 99))
    out["_diff"] = d - np.sum(w[masked] * d[masked]) / w[masked].sum()   # per-vertex field, for figures
    out["_masked"] = masked
    return out
