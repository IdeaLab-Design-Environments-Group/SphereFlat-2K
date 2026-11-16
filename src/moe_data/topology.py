"""Our own topology check for input meshes (we do not trust dataset metadata).

A mesh passes when it is a single closed, edge- and vertex-manifold, consistently
oriented surface with Euler characteristic 2 (a topological sphere). An inward
oriented sphere is not rejected: its faces are flipped and ``flipped_orientation``
is recorded. Unreferenced vertices are dropped; exactly coincident vertex
positions are *not* merged (they reject with ``duplicate_vertices``).

Everything is vectorised numpy/scipy so that ~1M-face meshes check in well under 2 s.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Reason codes, in the order the checks are applied.
REASON_CODES = (
    "degenerate_input",
    "duplicate_vertices",
    "not_closed",
    "edge_nonmanifold",
    "vertex_nonmanifold",
    "inconsistent_orientation",
    "multiple_components",
    "not_genus0",
)


def check_topology(V, F):
    """Return ``(ok, reason, measures)``; see :func:`check_and_orient`."""
    ok, reason, measures, _, _ = check_and_orient(V, F)
    return ok, reason, measures


def check_and_orient(V, F):
    """Check that (V, F) is a topological sphere and return it outward oriented.

    Returns ``(ok, reason, measures, V_out, F_out)``. ``V_out`` (float64) has the
    unreferenced vertices removed and ``F_out`` (int64) is re-indexed accordingly and
    flipped if the signed volume was negative. When ``ok`` is False, ``V_out`` and
    ``F_out`` are whatever was computed up to the failing check (may be None).
    ``measures`` holds JSON-able scalars only.
    """
    measures: dict = {}

    def fail(code, text, V_=None, F_=None):
        return False, f"{code}: {text}", measures, V_, F_

    # ---- degenerate input -------------------------------------------------------
    try:
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F)
    except (TypeError, ValueError) as e:
        return fail("degenerate_input", f"cannot convert arrays ({e})")
    measures["n_vertices"] = int(V.shape[0]) if V.ndim == 2 else 0
    measures["n_faces"] = int(F.shape[0]) if F.ndim == 2 else 0
    if V.ndim != 2 or V.shape[1] != 3:
        return fail("degenerate_input", f"V has shape {V.shape}, expected (n,3)")
    if F.ndim != 2 or F.shape[1] != 3:
        return fail("degenerate_input", f"F has shape {F.shape}, expected (m,3)")
    if F.shape[0] == 0:
        return fail("degenerate_input", "no faces")
    if not np.issubdtype(F.dtype, np.integer):
        return fail("degenerate_input", f"face indices have dtype {F.dtype}")
    F = F.astype(np.int64, copy=False)
    n = V.shape[0]
    if F.min() < 0 or F.max() >= n:
        return fail("degenerate_input", f"face index out of range [0,{n})")
    if not np.isfinite(V[np.unique(F)]).all():
        return fail("degenerate_input", "non-finite vertex coordinates")
    rep = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 2] == F[:, 0])
    if rep.any():
        return fail("degenerate_input", f"{int(rep.sum())} faces repeat a vertex index")

    # ---- drop unreferenced vertices --------------------------------------------
    used, F = np.unique(F, return_inverse=True)
    F = F.reshape(-1, 3).astype(np.int64)
    V = V[used]
    n = V.shape[0]
    m = F.shape[0]
    measures["n_unreferenced_removed"] = int(measures["n_vertices"] - n)
    measures["n_vertices"] = int(n)

    # ---- duplicate positions (not merged) --------------------------------------
    n_unique = np.unique(V + 0.0, axis=0).shape[0]  # +0.0 maps -0.0 to 0.0
    measures["n_duplicate_positions"] = int(n - n_unique)
    if n_unique < n:
        return fail("duplicate_vertices", f"{n - n_unique} vertices share an exact position", V, F)

    # ---- edges ----------------------------------------------------------------
    # half-edge k = 3*f + i goes F[f,i] -> F[f,(i+1)%3]
    hs = F.reshape(-1)
    ht = F[:, [1, 2, 0]].reshape(-1)
    lo = np.minimum(hs, ht)
    hi = np.maximum(hs, ht)
    ekey = lo * n + hi
    _, einv, ecount = np.unique(ekey, return_inverse=True, return_counts=True)
    n_edges = int(ecount.size)
    chi = int(n - n_edges + m)
    measures["n_edges"] = n_edges
    measures["chi"] = chi
    n_boundary = int((ecount == 1).sum())
    n_nonmanifold_e = int((ecount > 2).sum())
    measures["n_boundary_edges"] = n_boundary
    measures["n_nonmanifold_edges"] = n_nonmanifold_e

    # orientation: on a 2-face edge the two half-edges must run in opposite directions
    direction = np.where(hs < ht, 1, -1)
    dir_sum = np.bincount(einv, weights=direction, minlength=n_edges)
    n_inconsistent = int(((ecount == 2) & (dir_sum != 0)).sum())
    measures["n_inconsistent_edges"] = n_inconsistent

    # vertex manifoldness: count corner fans around each vertex (needs edges with <= 2 faces)
    n_nonmanifold_v = None
    if n_nonmanifold_e == 0:
        n_nonmanifold_v = _count_nonmanifold_vertices(hs, ht, einv, ecount, n, m)
    measures["n_nonmanifold_vertices"] = n_nonmanifold_v

    # connected components via shared edges
    adj = sp.coo_matrix((np.ones(hs.size, dtype=np.int8), (hs, ht)), shape=(n, n)).tocsr()
    n_comp, _ = connected_components(adj, directed=False)
    measures["n_components"] = int(n_comp)
    measures["genus"] = (2 - chi) // 2 if (n_comp == 1 and chi % 2 == 0) else None

    if n_boundary:
        return fail("not_closed", f"{n_boundary} boundary edges", V, F)
    if n_nonmanifold_e:
        return fail("edge_nonmanifold", f"{n_nonmanifold_e} edges with more than 2 faces", V, F)
    if n_nonmanifold_v:
        return fail("vertex_nonmanifold", f"{n_nonmanifold_v} vertices whose link is not a single cycle", V, F)
    if n_inconsistent:
        return fail("inconsistent_orientation", f"{n_inconsistent} edges traversed twice in the same direction", V, F)
    if n_comp != 1:
        return fail("multiple_components", f"{n_comp} connected components", V, F)
    if chi != 2:
        return fail("not_genus0", f"chi={chi} genus={measures['genus']}", V, F)

    # ---- outward orientation --------------------------------------------------
    vol = _signed_volume(V, F)
    measures["signed_volume"] = float(vol)
    measures["flipped_orientation"] = bool(vol < 0)
    if vol < 0:
        F = F[:, [0, 2, 1]]
    return True, "", measures, V, np.ascontiguousarray(F)


def _count_nonmanifold_vertices(hs, ht, einv, ecount, n, m):
    """Number of vertices whose incident faces form more than one edge-connected fan.

    Corners are nodes (corner 3f+i sits at vertex F[f,i]). Two corners at the same
    vertex are joined when their faces share an edge through that vertex. On a
    vertex-manifold surface each vertex has exactly one corner component.
    """
    order = np.argsort(einv, kind="stable")
    starts = np.concatenate(([0], np.cumsum(ecount)[:-1]))
    two = ecount == 2
    h1 = order[starts[two]]
    h2 = order[starts[two] + 1]
    s1, e1 = h1, 3 * (h1 // 3) + (h1 % 3 + 1) % 3  # corners at start / end of half-edge
    s2, e2 = h2, 3 * (h2 // 3) + (h2 % 3 + 1) % 3
    same_dir = hs[h1] == hs[h2]
    # corner of vertex hs[h1] in face 2 is s2 if same direction else e2
    a = np.concatenate([s1, e1])
    b = np.concatenate([np.where(same_dir, s2, e2), np.where(same_dir, e2, s2)])
    nc = 3 * m
    g = sp.coo_matrix((np.ones(a.size, dtype=np.int8), (a, b)), shape=(nc, nc)).tocsr()
    _, lab = connected_components(g, directed=False)
    corner_vertex = hs  # hs[k] is the vertex at corner k
    fans = np.unique(corner_vertex * nc + lab)
    fans_per_vertex = np.bincount(fans // nc, minlength=n)
    return int((fans_per_vertex > 1).sum())


def _signed_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)
