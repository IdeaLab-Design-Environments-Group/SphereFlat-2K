"""Dataset-level duplicate detection (runs between suitability and normalize).

Three relations link shapes:
  exact - equal :func:`exact_hash` (same triangles after rounding, any vertex/face order)
  near  - :func:`descriptor` distance below a candidate threshold and, when a confirmation
          test is given (run.py: :func:`chamfer` below a threshold), confirmed by it
  thing - same Thingi10K thing_id (sibling files of one upload)
Split groups are the connected components of the union of all three. Only exact/near
duplicates are rejected: within each exact/near component one representative is kept
(largest face count, ties by smallest id). Same-thing siblings are only grouped.
"""
import functools
import hashlib

import igl
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from moe_data.stages.normalize import normalize_mesh
from moe_data.stages.suitability import surface_stats

D2_BINS = 40
D2_MAX = 2.0          # distances on the unit-area mesh; larger ones go to the last bin
D2_PAIRS = 1_000_000  # histogram sampling noise ~ 1/sqrt(D2_PAIRS) in L2
D2_CHUNK = 250_000
# block weights: the D2 histogram sums to 1; extents and compactness are O(0.1) on
# unit-area shapes, so they enter with unit weight
W_HIST, W_EXTENT, W_COMPACT = 1.0, 1.0, 1.0

CHAMFER_SAMPLES = 20_000
CHAMFER_COARSE = 300       # samples per side when ranking the initial rotations
ICP_SCREEN = 16            # best-ranked initial rotations refined by a short ICP
ICP_SHORT = (5, 500)       # (iterations, samples per side) of the short ICP
ICP_STARTS = 2             # best after the short ICP, refined by the full ICP
ICP_FULL = (15, 2_000)
# principal axes whose extents are this close (smaller / larger) are not determined by PCA;
# rotations about the remaining axis (one degenerate pair) or over SO(3) (all three) are tried
DEGENERATE_RATIO = 0.8
PLANE_STEP_DEG = 5.0
N_SO3 = 1_000              # fixed random rotations used when all three extents are degenerate
# rotations that map the principal frame onto itself: identity and the three axis pairs flipped
PROPER_FLIPS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)


def canonical_faces(V, F, decimals: int = 9):
    """Faces reordered by geometry alone: each face rotated so its lexicographically smallest
    corner comes first, faces sorted by those corner coordinates (rounded to ``decimals``, so
    rounding-level differences in V do not reorder ties). Random sampling with a fixed seed then
    does not depend on the input face order or vertex numbering."""
    F = np.asarray(F, dtype=np.int64)
    T = np.round(np.asarray(V, dtype=np.float64), decimals)[F]
    r = _lexmin_corner(T)
    idx = (r[:, None] + np.arange(3)[None, :]) % 3
    F = np.take_along_axis(F, idx, axis=1)
    T = np.take_along_axis(T, idx[:, :, None], axis=1).reshape(-1, 9)
    return F[np.lexsort(T.T[::-1])]


def sample_surface(V, F, k, rng):
    """k area-weighted uniform random points on the surface."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    t = rng.choice(F.shape[0], size=k, p=areas / areas.sum())
    u, v = rng.random(k), rng.random(k)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[t] + u[:, None] * (b[t] - a[t]) + v[:, None] * (c[t] - a[t])


def descriptor(V, F, rng) -> np.ndarray:
    """Shape descriptor of the unit-area, centred mesh (normalised internally).

    [W_HIST * D2 histogram (D2_BINS, sums to 1), W_EXTENT * sorted PCA extents (3),
     W_COMPACT * compactness]. Faces are put in canonical order before sampling, so the
    result does not depend on face order. Pass a freshly seeded rng per shape for
    reproducibility.
    """
    Vn, _, _, _ = normalize_mesh(V, F)
    F = canonical_faces(Vn, F)
    counts = np.zeros(D2_BINS, dtype=np.int64)
    for start in range(0, D2_PAIRS, D2_CHUNK):
        k = min(D2_CHUNK, D2_PAIRS - start)
        p = sample_surface(Vn, F, 2 * k, rng)
        d = np.linalg.norm(p[:k] - p[k:], axis=1)
        counts += np.bincount(np.minimum((d / D2_MAX * D2_BINS).astype(np.int64), D2_BINS - 1),
                              minlength=D2_BINS)
    hist = counts / D2_PAIRS
    st = surface_stats(Vn, F)
    compact = abs(st["volume"]) / st["area"] ** 1.5
    return np.concatenate([W_HIST * hist, W_EXTENT * st["extents"], [W_COMPACT * compact]])


def _principal_frame(V, F):
    """(eigenvectors as columns, ascending, det +1; extents = sqrt of the eigenvalues) of the exact
    surface covariance of a centred mesh."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    s = a + b + c
    second = (np.einsum("t,ti,tj->ij", areas, a, a) + np.einsum("t,ti,tj->ij", areas, b, b)
              + np.einsum("t,ti,tj->ij", areas, c, c) + np.einsum("t,ti,tj->ij", areas, s, s)) / 12.0
    w, E = np.linalg.eigh(0.5 * (second + second.T) / areas.sum())
    if np.linalg.det(E) < 0:
        E[:, 0] = -E[:, 0]
    return E, np.sqrt(np.clip(w, 0.0, None))


def aligned_shape(V, F, rng) -> dict:
    """Unit-area, centred mesh rotated into its principal frame, with CHAMFER_SAMPLES
    area-weighted surface samples (canonical face order, so the samples depend only on the
    geometry and the rng). Input for :func:`chamfer`."""
    Vn, _, _, _ = normalize_mesh(V, F)
    F = canonical_faces(Vn, F)
    E, ext = _principal_frame(Vn, F)
    Va = np.ascontiguousarray(Vn @ E)
    return {"V": Va, "F": F, "P": sample_surface(Va, F, CHAMFER_SAMPLES, rng), "extents": ext}


def _distance(shape, P, exact=True):
    """Distances from points P to the shape's surface and the closest surface points. With
    ``exact=False`` the surface is replaced by the shape's samples (nearest sample point): biased
    by the sample spacing but faster on large meshes; used only to rank and pre-align motions."""
    if not exact:
        if "kdtree" not in shape:
            shape["kdtree"] = cKDTree(shape["P"])
        d, i = shape["kdtree"].query(P)
        return d, shape["P"][i]
    if "aabb" not in shape:
        shape["aabb"] = igl.AABB()
        shape["aabb"].init(shape["V"], shape["F"])
    d2, _, C = shape["aabb"].squared_distance(shape["V"], shape["F"], np.ascontiguousarray(P))
    return np.sqrt(d2), C


def _axis_rotation(axis, deg):
    t = np.deg2rad(deg)
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(3)
    R[i, i] = R[j, j] = np.cos(t)
    R[i, j], R[j, i] = -np.sin(t), np.sin(t)
    return R


@functools.lru_cache(maxsize=1)
def _so3_rotations():
    return Rotation.random(N_SO3, random_state=0).as_matrix()


def _initial_rotations(a, b):
    """Rotations (a's principal frame -> b's) to start from: the four proper flips; when either
    shape has two principal extents within DEGENERATE_RATIO, those flips combined with rotations
    about the remaining axis in PLANE_STEP_DEG steps; when all three are, also N_SO3 fixed
    random rotations."""
    degenerate = set()
    for ext in (a["extents"], b["extents"]):
        for i in (0, 1):
            if ext[i] >= DEGENERATE_RATIO * ext[i + 1]:
                degenerate.add(i)
    axes = sorted(2 if i == 0 else 0 for i in degenerate)  # rotate about the axis outside the pair
    turns = [np.eye(3)] + [_axis_rotation(ax, k * PLANE_STEP_DEG) for ax in axes
                           for k in range(1, int(round(360.0 / PLANE_STEP_DEG)))]
    out = [T @ np.diag(s) for s in PROPER_FLIPS for T in turns]
    if len(degenerate) == 2:
        out += list(_so3_rotations())
    return out


def _sym_mean(a, b, R, t, k, exact=True):
    """0.5 * (mean distance of a's first k samples, mapped x -> R x + t, to b's surface + mean
    distance of b's first k samples, mapped back, to a's surface), for a batch of motions:
    R (m, 3, 3), t (m, 3) -> (m,)."""
    m = len(R)
    Pa, Pb = a["P"][:k], b["P"][:k]
    da, _ = _distance(b, (np.einsum("mij,kj->mki", R, Pa) + t[:, None, :]).reshape(-1, 3), exact)
    db, _ = _distance(a, np.einsum("mji,mkj->mki", R, Pb[None, :, :] - t[:, None, :]).reshape(-1, 3), exact)
    return 0.5 * (da.reshape(m, -1).mean(1) + db.reshape(m, -1).mean(1))


def _kabsch(X, Y):
    """Proper rigid motions (R, t) minimising sum |R x + t - y|^2, batched: X, Y (m, n, 3)."""
    mx, my = X.mean(1), Y.mean(1)
    H = np.einsum("mni,mnj->mij", X - mx[:, None], Y - my[:, None])
    U, _, Wt = np.linalg.svd(H)
    V = np.transpose(Wt, (0, 2, 1))
    D = np.tile(np.eye(3), (len(X), 1, 1))
    D[:, 2, 2] = np.sign(np.linalg.det(V @ np.transpose(U, (0, 2, 1))))
    R = V @ D @ np.transpose(U, (0, 2, 1))
    return R, my - np.einsum("mij,mj->mi", R, mx)


def _icp(a, b, R, t, iters, k, exact=True):
    """Symmetric point-to-surface ICP on k samples per side, for a batch of start motions. The
    query samples are a["P"][:k] / b["P"][:k]; with ``exact=False`` the targets are the other
    shape's full sample set."""
    m = len(R)
    Pa, Pb = a["P"][:k], b["P"][:k]
    for _ in range(iters):
        # a's samples -> closest points on b; b's samples (mapped back) -> closest points on a
        _, Ca = _distance(b, (np.einsum("mij,kj->mki", R, Pa) + t[:, None, :]).reshape(-1, 3), exact)
        _, Cb = _distance(a, np.einsum("mji,mkj->mki", R, Pb[None, :, :] - t[:, None, :]).reshape(-1, 3), exact)
        X = np.concatenate([np.broadcast_to(Pa, (m, k, 3)), Cb.reshape(m, k, 3)], axis=1)
        Y = np.concatenate([Ca.reshape(m, k, 3), np.broadcast_to(Pb, (m, k, 3))], axis=1)
        R, t = _kabsch(X, Y)
    return R, t


def chamfer(a: dict, b: dict) -> float:
    """Symmetric point-to-surface Chamfer distance between two :func:`aligned_shape` results
    (unit-area length units): 0.5 * (mean distance of a's samples to b's surface + mean distance
    of b's samples to a's surface), minimised over proper rigid motions of a.

    Initial rotations (:func:`_initial_rotations`) are ranked on CHAMFER_COARSE samples and the
    ICP_SCREEN best get a short ICP, both against the other shape's sample cloud; the ICP_STARTS
    best of those get a full ICP against the surface, and the result is evaluated on all samples. Point-to-surface distances make the value ~0 for identical
    surfaces whatever the samples.
    """
    R = np.asarray(_initial_rotations(a, b))
    t = np.zeros((len(R), 3))
    pick = np.argsort(_sym_mean(a, b, R, t, CHAMFER_COARSE, exact=False), kind="stable")[:ICP_SCREEN]
    R, t = _icp(a, b, R[pick], t[pick], *ICP_SHORT, exact=False)
    pick = np.argsort(_sym_mean(a, b, R, t, ICP_SHORT[1], exact=False), kind="stable")[:ICP_STARTS]
    R, t = _icp(a, b, R[pick], t[pick], *ICP_FULL)
    return float(_sym_mean(a, b, R, t, len(a["P"])).min())


def exact_hash(V, F, decimals: int) -> str:
    """Hash of the set of triangles with coordinates rounded to ``decimals``.

    Invariant to vertex order, face order and cyclic rotation of each face (orientation
    is kept). Coordinates are used as given: normalise first for translation/scale
    invariance.
    """
    q = np.rint(np.asarray(V, dtype=np.float64) * 10.0**decimals).astype(np.int64)
    T = q[np.asarray(F, dtype=np.int64)]                      # (m,3,3)
    # rotate each face so its lexicographically smallest corner comes first
    r = _lexmin_corner(T)
    idx = (r[:, None] + np.arange(3)[None, :]) % 3
    T = np.take_along_axis(T, idx[:, :, None], axis=1).reshape(-1, 9)
    T = T[np.lexsort(T.T[::-1])]
    return hashlib.sha1(np.ascontiguousarray(T).tobytes()).hexdigest()


def _lexmin_corner(T):
    """Index (0..2) of the lexicographically smallest corner of each face."""
    best = np.zeros(T.shape[0], dtype=np.int64)
    for i in (1, 2):
        cur, cand = T[np.arange(T.shape[0]), best], T[:, i]
        diff = cand - cur
        nz = diff != 0
        first = np.argmax(nz, axis=1)
        less = nz.any(axis=1) & (diff[np.arange(T.shape[0]), first] < 0)
        best[less] = i
    return best


def _components(n, pairs):
    if len(pairs) == 0:
        return np.arange(n)
    p = np.asarray(pairs, dtype=np.int64)
    g = sp.coo_matrix((np.ones(len(p), dtype=np.int8), (p[:, 0], p[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _relations(descriptors, hashes, thing_ids, threshold, confirm=None):
    n = len(hashes)
    rel = {"exact": [], "near": [], "thing": []}
    for name, keys in (("exact", hashes), ("thing", thing_ids)):
        first = {}
        for i, h in enumerate(keys):
            if h is None:
                continue
            if h in first:
                rel[name].append((first[h], i))
            else:
                first[h] = i
    if n > 1:
        D = np.asarray(descriptors, dtype=np.float64).reshape(n, -1)
        cand = cKDTree(D).query_pairs(r=threshold, output_type="ndarray")
        if len(cand):
            cand = cand[np.lexsort((cand[:, 1], cand[:, 0]))]
            cand = cand[np.linalg.norm(D[cand[:, 0]] - D[cand[:, 1]], axis=1) < threshold]
        if len(cand) and confirm is not None:
            cand = cand[np.asarray(confirm(cand), dtype=bool)]
        rel["near"] = [tuple(int(v) for v in x) for x in cand]
    return rel


def dedup_records(ids, descriptors, hashes, thing_ids, threshold, n_faces, confirm=None) -> dict:
    """Per id: ``group_id`` (split group), ``dup_group_reason`` ('exact', 'near', 'thing'
    joined by '+', the relations linking this member; '' if alone), ``keep`` and, for
    rejected members, ``reject_stage='dedup'`` and ``reject_reason='duplicate_of: <id>'``.

    Pairs with descriptor distance below ``threshold`` are near-duplicate candidates. With
    ``confirm`` (callable: (k, 2) array of index pairs -> k bools) only the confirmed candidates
    are near duplicates; without it every candidate is.
    """
    ids = list(ids)
    n = len(ids)
    rel = _relations(descriptors, hashes, thing_ids, threshold, confirm)
    member = [set() for _ in range(n)]
    for name, pairs in rel.items():
        for i, j in pairs:
            member[i].add(name)
            member[j].add(name)
    group = _components(n, rel["exact"] + rel["near"] + rel["thing"])
    dup = _components(n, rel["exact"] + rel["near"])

    group_name = {}
    for i in sorted(range(n), key=lambda i: str(ids[i])):
        group_name.setdefault(group[i], str(ids[i]))      # smallest id names the group
    rep = {}
    for i in range(n):
        key = (-int(n_faces[i]), str(ids[i]))
        if dup[i] not in rep or key < rep[dup[i]][0]:
            rep[dup[i]] = (key, i)
    out = {}
    for i in range(n):
        r = rep[dup[i]][1]
        rec = {"group_id": group_name[group[i]],
               "dup_group_reason": "+".join(sorted(member[i])),
               "keep": r == i}
        if r != i:
            rec["reject_stage"] = "dedup"
            rec["reject_reason"] = f"duplicate_of: {ids[r]}"
        out[ids[i]] = rec
    return out


def group_duplicates(ids, descriptors, hashes, thing_ids, threshold, n_faces=None, confirm=None) -> dict:
    """id -> group_id (connected components of exact ∪ near ∪ same thing_id)."""
    nf = n_faces if n_faces is not None else [0] * len(list(ids))
    recs = dedup_records(ids, descriptors, hashes, thing_ids, threshold, nf, confirm)
    return {k: v["group_id"] for k, v in recs.items()}
