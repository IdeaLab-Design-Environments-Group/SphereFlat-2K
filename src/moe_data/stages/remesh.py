"""Remesh stage: isotropic explicit remeshing to ~target_vertices, then re-checks.

Input is the normalised mesh (area-weighted centroid 0, total area 1). Steps:

1. Isotropic explicit remeshing (pymeshlab) with an absolute target edge length h. For a
   closed mesh of area A made of near-equilateral triangles of side h, F ~ 2 V and each
   triangle has area sqrt(3)/4 h^2, so V ~ 2 A / (sqrt(3) h^2), i.e.
   h = sqrt(2 A / (sqrt(3) * target_vertices)). Remeshing a unit-area icosphere with this h
   gives ~14.1k vertices for a 15k target; h is rescaled by sqrt(n / target) and the remesh
   redone from the input until n is in [min_vertices, max_vertices] (max_attempts).
2. Unreferenced vertices and the topology check (closed, manifold, oriented, chi = 2).
3. Area ratio A_remesh / A_orig, recorded before rescaling.
4. Rescale to unit area and recentre (same normalisation as the normalize stage).
5. Symmetric Hausdorff distance between the final mesh and the input, relative to the input
   bounding-box diagonal. Both directed distances are recorded; the orig->remesh direction is
   the one that sees lost thin features.
6. Self-intersections of the remesh (recorded, not a reject) and triangle quality (zero-area
   triangles reject; minimum angle recorded).
"""
import numpy as np
import pymeshlab

from moe_data.stages.base import StageResult
from moe_data.stages.normalize import normalize_mesh

NAME = "remesh"

# Hausdorff samples per face of the sampled mesh (in addition to every vertex). Not a
# threshold: it only sets the sampling density; overridable with cfg['remesh']['hausdorff_samples_per_face'].
_DEFAULT_SAMPLES_PER_FACE = 4


def target_edge_length(area: float, n_vertices: float) -> float:
    """Edge length h for which an equilateral triangulation of `area` has ~n_vertices vertices."""
    return float(np.sqrt(2.0 * area / (np.sqrt(3.0) * n_vertices)))


def isotropic_remesh(V, F, h: float, iterations: int):
    """One pymeshlab isotropic explicit remeshing pass with absolute target length h."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(V, dtype=np.float64),
                               face_matrix=np.asarray(F, dtype=np.int32)))
    ms.meshing_isotropic_explicit_remeshing(iterations=int(iterations),
                                            targetlen=pymeshlab.PureValue(float(h)))
    m = ms.current_mesh()
    return (np.asarray(m.vertex_matrix(), dtype=np.float64),
            np.asarray(m.face_matrix(), dtype=np.int64))


def directed_hausdorff(V_from, F_from, V_to, F_to, samples_per_face: int = _DEFAULT_SAMPLES_PER_FACE):
    """Max distance from points sampled on (V_from, F_from) to the surface (V_to, F_to).

    Samples every vertex plus samples_per_face * n_faces points on the faces. maxdist is set far
    above any possible distance (pymeshlab ignores samples farther than maxdist).
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(V_from, dtype=np.float64), np.asarray(F_from, dtype=np.int32)))
    ms.add_mesh(pymeshlab.Mesh(np.asarray(V_to, dtype=np.float64), np.asarray(F_to, dtype=np.int32)))
    ext = np.ptp(np.vstack([V_from, V_to]), axis=0)
    r = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1, savesample=False,
                                  samplevert=True, sampleedge=False, samplefauxedge=False,
                                  sampleface=True, samplenum=int(samples_per_face * len(F_from)),
                                  maxdist=pymeshlab.PureValue(float(10.0 * np.linalg.norm(ext) + 1.0)))
    return float(r["max"])


def count_self_intersecting_faces(V, F) -> int:
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    return int(ms.current_mesh().selected_face_number())


def triangle_quality(V, F):
    """(areas (m,), min corner angle in degrees over all triangles)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    angles = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        u, w = q - p, r - p
        cosang = np.einsum("ij,ij->i", u, w) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1))
        angles.append(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    return areas, float(np.min(angles))


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    from moe_data.topology import check_topology  # lazy: topology module owned elsewhere

    V0, F0 = inputs.get("V"), inputs.get("F")
    if V0 is None or F0 is None:
        return StageResult(False, reason="missing_input: V or F not in inputs")
    V0 = np.asarray(V0, dtype=np.float64)
    F0 = np.asarray(F0, dtype=np.int64)
    rc = cfg["remesh"]
    target, lo, hi = int(rc["target_vertices"]), int(rc["min_vertices"]), int(rc["max_vertices"])
    spf = int(rc.get("hausdorff_samples_per_face", _DEFAULT_SAMPLES_PER_FACE))
    meta: dict = {}

    a0, b0, c0 = V0[F0[:, 0]], V0[F0[:, 1]], V0[F0[:, 2]]
    area0 = float(0.5 * np.linalg.norm(np.cross(b0 - a0, c0 - a0), axis=1).sum())
    if not (np.isfinite(V0).all() and area0 > 0 and np.isfinite(area0)):
        return StageResult(False, meta={"orig_area": area0}, reason="remesh_bad_input: non-finite or zero area")

    # ---- remesh, adjusting h until the vertex count is in range ---------------------------
    # n(h) is monotone but stepwise, so the sqrt(n/target) update can oscillate on a narrow
    # range; once h is bracketed (one h gave too many vertices, one too few) bisect geometrically.
    h = target_edge_length(area0, target)
    h_too_small, h_too_large = None, None
    n = 0
    for attempt in range(1, int(rc["max_attempts"]) + 1):
        V, F = isotropic_remesh(V0, F0, h, rc["iterations"])
        n = len(V)
        meta.update(remesh_attempts=attempt, remesh_target_len=h, remesh_n_vertices=n,
                    remesh_n_faces=len(F))
        if lo <= n <= hi or n == 0:
            break
        if n > hi:
            h_too_small = h if h_too_small is None else max(h_too_small, h)
        else:
            h_too_large = h if h_too_large is None else min(h_too_large, h)
        if h_too_small is not None and h_too_large is not None:
            h = float(np.sqrt(h_too_small * h_too_large))
        else:
            h *= np.sqrt(n / target)
    if not (lo <= n <= hi):
        return StageResult(False, meta=meta,
                           reason=f"remesh_vertex_count: {n} not in [{lo},{hi}] after {meta['remesh_attempts']} attempts")

    # ---- unreferenced vertices and topology ------------------------------------------------
    n_unref = int(n - np.unique(F).size)
    meta["remesh_n_unreferenced"] = n_unref
    if n_unref:
        return StageResult(False, meta=meta, reason=f"remesh_unreferenced: {n_unref} unreferenced vertices")
    ok, reason, measures = check_topology(V, F)
    meta.update({f"remesh_{k}": v for k, v in measures.items()})
    if not ok:
        return StageResult(False, meta=meta, reason=f"remesh_topology: {reason}")
    if measures.get("flipped_orientation"):
        return StageResult(False, meta=meta, reason="remesh_topology: remesh is inward oriented")

    # ---- area ratio (before rescaling), then normalise ------------------------------------
    Vn, area_r, _, _ = normalize_mesh(V, F)
    meta["remesh_area_ratio"] = float(area_r / area0)
    if Vn is None:
        return StageResult(False, meta=meta, reason=f"remesh_zero_area: area={area_r:.3g}")
    V = Vn

    # ---- symmetric Hausdorff on the final normalised mesh vs the normalised input ----------
    diag = float(np.linalg.norm(np.ptp(V0, axis=0)))
    d_rem_to_orig = directed_hausdorff(V, F, V0, F0, spf) / diag
    d_orig_to_rem = directed_hausdorff(V0, F0, V, F, spf) / diag
    d_sym = max(d_rem_to_orig, d_orig_to_rem)
    meta.update(hausdorff_remesh_to_orig=d_rem_to_orig, hausdorff_orig_to_remesh=d_orig_to_rem,
                hausdorff_sym=d_sym)
    tol = float(rc["hausdorff_tol"])
    if not (d_sym <= tol):  # accept-if form: NaN rejects
        return StageResult(False, meta=meta, reason=f"remesh_hausdorff: {d_sym:.3g} > {tol:.3g} of bbox diagonal")

    # ---- self-intersections (flag) and triangle quality ------------------------------------
    n_si = count_self_intersecting_faces(V, F)
    meta["remesh_n_self_intersecting_faces"] = n_si
    meta["self_intersecting_remesh"] = bool(n_si > 0)
    areas, min_angle = triangle_quality(V, F)
    meta["remesh_min_tri_area"] = float(areas.min())
    meta["remesh_min_angle_deg"] = min_angle
    if not (areas.min() > 0):
        return StageResult(False, meta=meta,
                           reason=f"remesh_degenerate: {int((~(areas > 0)).sum())} zero-area triangles")

    return StageResult(True, data={"V": V, "F": F}, meta=meta)
