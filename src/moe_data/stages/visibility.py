"""visibility stage: synthetic per-vertex weight fields in [0, 1] on ``V_sub``.

There is no real data on which parts of an object are seen, so two kinds of field are synthesised:

* ``view``: a perspective pinhole camera at ``centre + camera_distance * R_bbox * d`` (d uniform on S^2).
  A vertex is visible if it is front-facing, n.(c - x) > 0, and the ray from x (pushed off the surface by
  ``ray_offset_rel * R_bbox`` along n) towards c hits no triangle other than the vertex's own 1-ring.
  The binary mask is softened by one implicit heat step (M - t L) w = M w0 (cotan L, igl sign: negative semi-definite) and clipped to [0, 1].
* ``paint``: 1..4 blobs; seeds drawn area-weighted, each blob one implicit heat step from the seed
  (geodesic-like kernel) divided by its peak, blobs combined with max.

Ray engine: trimesh 5.2.1 without embree (rtree candidates, one Python query per ray) took 22 s for one
view on a 31k-vertex protrusion mesh. Every ray of a pinhole camera passes through the pinhole, so a ray
is a single point in the image plane: triangles are projected, bucketed in a 2D grid, and each vertex is
tested (vectorised Moeller-Trumbore on the camera->origin segment) only against triangles in its cell.
This is exact, not a depth buffer; tests check it against trimesh's ``intersects_any``.

The rng is SeedSequence([seed, crc32(shape_id), crc32("visibility")]), so reruns give identical fields.
"""
import time
import zlib

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from moe_data import geometry as geo
from moe_data.stages.base import StageResult

NAME = "visibility"


def make_rng(seed: int, shape_id: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(
        [int(seed), zlib.crc32(shape_id.encode()), zlib.crc32(NAME.encode())]))


def random_direction(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _heat_solver(V: np.ndarray, F: np.ndarray, t: float):
    """Factorised implicit heat step: returns (solve, M) with solve(b) = (M - t L)^-1 b."""
    M = sp.diags(geo.vertex_areas_barycentric(V, F))
    lu = spla.splu((M - t * geo.cotan_laplacian(V, F)).tocsc())
    return lu.solve, M


def _camera_frame(d: np.ndarray) -> np.ndarray:
    """Rows: two image-plane axes and the viewing axis -d."""
    w = -np.asarray(d, np.float64) / np.linalg.norm(d)
    a = np.eye(3)[np.argmin(np.abs(w))]
    u = np.cross(w, a)
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(w, u), w])


def _ranges(starts: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(owner, value): for every k, values starts[k] .. starts[k]+counts[k]-1 tagged with owner k."""
    owner = np.repeat(np.arange(len(counts)), counts)
    first = np.cumsum(counts) - counts
    return owner, starts[owner] + np.arange(owner.size) - first[owner]


def occluded(V: np.ndarray, F: np.ndarray, cam: np.ndarray, origins: np.ndarray,
             query: np.ndarray) -> np.ndarray:
    """(k,) bool: does the segment origins[i] -> cam cross a triangle not incident to vertex query[i]?

    The camera must lie outside the mesh's bounding sphere seen along its axis (all depths > 0).
    """
    R = _camera_frame(cam - V.mean(axis=0))
    rel = V - cam
    z = rel @ R[2]
    p = (rel @ R[:2].T) / z[:, None]
    ro = origins - cam
    q = (ro @ R[:2].T) / (ro @ R[2])[:, None]

    tp = p[F]                                                     # (m,3,2) projected triangles
    lo, hi = tp.min(axis=1), tp.max(axis=1)
    h = max(float(np.median((hi - lo).max(axis=1))), 1e-12)      # cell ~ a typical triangle
    base = np.minimum(lo.min(axis=0), q.min(axis=0))
    i0 = np.floor((lo - base) / h).astype(np.int64)
    i1 = np.floor((hi - base) / h).astype(np.int64)
    ny = int(max(i1[:, 1].max(), np.floor((q[:, 1] - base[1]) / h).max())) + 1
    span = i1 - i0 + 1
    tri, k = _ranges(np.zeros(len(F), np.int64), span[:, 0] * span[:, 1])
    cell = (i0[tri, 0] + k // span[tri, 1]) * ny + i0[tri, 1] + k % span[tri, 1]
    order = np.argsort(cell, kind="stable")
    cell, tri = cell[order], tri[order]

    qi = np.floor((q - base) / h).astype(np.int64)
    qcell = qi[:, 0] * ny + qi[:, 1]
    start = np.searchsorted(cell, qcell, "left")
    qq, pos = _ranges(start, np.searchsorted(cell, qcell, "right") - start)
    t = tri[pos]
    keep = ~np.any(F[t] == query[qq, None], axis=1)              # the vertex's own 1-ring never occludes it
    qq, t = qq[keep], t[keep]

    # Moeller-Trumbore, ray cam + s*(origin - cam), hit if 0 < s < 1
    a, b, c = V[F[t, 0]], V[F[t, 1]], V[F[t, 2]]
    e1, e2, dirn = b - a, c - a, ro[qq]
    pv = np.cross(dirn, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-300
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = cam - a
    bu = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    bv = np.einsum("ij,ij->i", dirn, qv) * inv
    s = np.einsum("ij,ij->i", e2, qv) * inv
    hit = ok & (bu >= 0) & (bv >= 0) & (bu + bv <= 1) & (s > 0) & (s < 1)
    return np.bincount(qq[hit], minlength=len(query)) > 0


def visible_mask(V: np.ndarray, F: np.ndarray, d: np.ndarray, cfg_vis: dict,
                 normals: np.ndarray | None = None) -> np.ndarray:
    """Binary visibility (n,) bool from a perspective camera along unit direction d."""
    lo, hi = V.min(axis=0), V.max(axis=0)
    centre, radius = 0.5 * (lo + hi), 0.5 * np.linalg.norm(hi - lo)
    cam = centre + cfg_vis["camera_distance"] * radius * np.asarray(d, np.float64)
    N = geo.vertex_normals(V, F) if normals is None else normals
    front = np.einsum("ij,ij->i", N, cam - V) > 0
    vis = np.zeros(len(V), bool)
    idx = np.flatnonzero(front)
    if idx.size:
        origins = V[idx] + cfg_vis["ray_offset_rel"] * radius * N[idx]
        vis[idx] = ~occluded(V, F, cam, origins, idx)
    return vis


def paint_field(solve, M: sp.dia_matrix, seeds: np.ndarray) -> np.ndarray:
    """max over seeds of the peak-normalised heat kernel; values in [0, 1], 1 at each seed."""
    n = M.shape[0]
    B = np.zeros((n, len(seeds)))
    B[seeds, np.arange(len(seeds))] = M.diagonal()[seeds]
    U = solve(B).reshape(n, len(seeds))
    U = U / U.max(axis=0, keepdims=True)
    return np.clip(U.max(axis=1), 0.0, 1.0)


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    missing = [k for k in ("V_sub", "F") if inputs.get(k) is None]
    if missing:
        return StageResult(False, reason=f"missing_input: {', '.join(missing)}")
    V = np.asarray(inputs["V_sub"], np.float64)
    F = np.asarray(inputs["F"], np.int64)
    cv = cfg["visibility"]
    n_view, n_paint = int(cv["n_view"]), int(cv["n_paint"])
    rng = make_rng(cfg["seed"], shape_id)
    area = geo.vertex_areas_barycentric(V, F)
    N = geo.vertex_normals(V, F)

    fields, kinds, dirs, frac_vis, t_ray = [], [], [], [], 0.0
    view_solve, M = _heat_solver(V, F, float(cv["view_smooth_t"]))
    for _ in range(n_view):
        d = random_direction(rng)
        t0 = time.perf_counter()
        mask = visible_mask(V, F, d, cv, N)
        t_ray += time.perf_counter() - t0
        w = np.clip(view_solve(M @ mask.astype(np.float64)), 0.0, 1.0)
        fields.append(w)
        kinds.append("view")
        dirs.append(d)
        frac_vis.append(float(area[mask].sum() / area.sum()))

    paint_solve, M = _heat_solver(V, F, float(cv["paint_diffusion_t"]))
    lo, hi = cv["paint_blobs"]
    paint_seeds = []
    for _ in range(n_paint):
        k = int(rng.integers(lo, hi + 1))
        seeds = rng.choice(len(V), size=k, replace=False, p=area / area.sum())
        fields.append(paint_field(paint_solve, M, seeds))
        kinds.append("paint")
        paint_seeds.append([int(s) for s in seeds])

    vis = np.stack(fields, axis=1).astype(np.float32) if fields else np.zeros((len(V), 0), np.float32)
    vis_dirs = np.array(dirs, np.float64).reshape(n_view, 3)
    meta = {"vis_mean": [float(m) for m in vis.mean(axis=0)], "vis_frac_visible": frac_vis,
            "vis_raycast_s": t_ray, "vis_dirs": vis_dirs.tolist(), "vis_paint_seeds": paint_seeds,
            "vis_view_frac_ok": bool(all(0.05 < f < 0.95 for f in frac_vis))}
    if not np.all(np.isfinite(vis)):
        return StageResult(False, meta=meta, reason="nonfinite_vis: heat solve produced NaN/inf")
    if vis.size and (vis.min() < 0.0 or vis.max() > 1.0):
        return StageResult(False, meta=meta, reason="vis_out_of_range: weight outside [0, 1]")
    return StageResult(True, data={"vis": vis, "vis_kind": np.array(kinds, dtype="U5"), "vis_dirs": vis_dirs},
                       meta=meta)
