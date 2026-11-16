"""Dataset-level duplicate detection (runs between suitability and normalize).

Three relations link shapes:
  exact - equal :func:`exact_hash` (same triangles after rounding, any vertex/face order)
  near  - L2 distance between :func:`descriptor` vectors below a threshold
  thing - same Thingi10K thing_id (sibling files of one upload)
Split groups are the connected components of the union of all three. Only exact/near
duplicates are rejected: within each exact/near component one representative is kept
(largest face count, ties by smallest id). Same-thing siblings are only grouped.
"""
import hashlib

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from moe_data.stages.normalize import normalize_mesh
from moe_data.stages.suitability import surface_stats

D2_BINS = 40
D2_MAX = 2.0          # distances on the unit-area mesh; larger ones go to the last bin
D2_PAIRS = 100_000
# block weights: the D2 histogram sums to 1; extents and compactness are O(0.1) on
# unit-area shapes, so they enter with unit weight
W_HIST, W_EXTENT, W_COMPACT = 1.0, 1.0, 1.0


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
     W_COMPACT * compactness]. Pass a freshly seeded rng per shape for reproducibility.
    """
    Vn, _, _, _ = normalize_mesh(V, F)
    F = np.asarray(F, dtype=np.int64)
    p = sample_surface(Vn, F, 2 * D2_PAIRS, rng)
    d = np.linalg.norm(p[:D2_PAIRS] - p[D2_PAIRS:], axis=1)
    hist = np.bincount(np.minimum((d / D2_MAX * D2_BINS).astype(np.int64), D2_BINS - 1),
                       minlength=D2_BINS) / D2_PAIRS
    st = surface_stats(Vn, F)
    compact = abs(st["volume"]) / st["area"] ** 1.5
    return np.concatenate([W_HIST * hist, W_EXTENT * st["extents"], [W_COMPACT * compact]])


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


def _relations(descriptors, hashes, thing_ids, threshold):
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
            dist = np.linalg.norm(D[cand[:, 0]] - D[cand[:, 1]], axis=1)
            rel["near"] = [tuple(x) for x in cand[dist < threshold]]
    return rel


def dedup_records(ids, descriptors, hashes, thing_ids, threshold, n_faces) -> dict:
    """Per id: ``group_id`` (split group), ``dup_group_reason`` ('exact', 'near', 'thing'
    joined by '+', the relations linking this member; '' if alone), ``keep`` and, for
    rejected members, ``reject_stage='dedup'`` and ``reject_reason='duplicate_of: <id>'``.
    """
    ids = list(ids)
    n = len(ids)
    rel = _relations(descriptors, hashes, thing_ids, threshold)
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


def group_duplicates(ids, descriptors, hashes, thing_ids, threshold, n_faces=None) -> dict:
    """id -> group_id (connected components of exact ∪ near ∪ same thing_id)."""
    nf = n_faces if n_faces is not None else [0] * len(list(ids))
    recs = dedup_records(ids, descriptors, hashes, thing_ids, threshold, nf)
    return {k: v["group_id"] for k, v in recs.items()}
