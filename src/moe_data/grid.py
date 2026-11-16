"""Lat-long grid channels on the unit sphere.

Conventions (docs/CONTRACTS.md):

* row i has colatitude theta_i = (i + 0.5) * pi / H (row 0 next to the north pole, z = +1);
* column j has longitude phi_j = (j + 0.5) * 2 pi / W - pi;
* a point is x = sin(theta) cos(phi), y = sin(theta) sin(phi), z = cos(theta);
* a rotated sphere map is P' = P @ R.T (p' = R p), and a field f on the original sphere becomes
  f_R(x) = f(R^T x) on the rotated sphere. ``compute_grid(..., R=R)`` evaluates every channel at R^T c on
  the original float64 P, so the grid is in the same frame as ``P @ R.T`` (exact resampling, option a).

Pure NumPy/SciPy except ``rotate_grid_torch`` (option b, the interpolation alternative kept for comparison).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

CHANNELS = ("psi", "psi_min", "K_pt", "H", "K_int_density", "K_int_density_asinh", "vertex_density")

# ----------------------------------------------------------------------------------------- lat-long


def latlong_cell_centers(H: int, W: int) -> np.ndarray:
    """Unit cell-centre directions, shape (H*W, 3), flat index = i*W + j."""
    theta = (np.arange(H) + 0.5) * np.pi / H
    phi = (np.arange(W) + 0.5) * 2 * np.pi / W - np.pi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)


def latlong_cell_solid_angle(H: int, W: int) -> np.ndarray:
    """Exact solid angle of one cell in each row, shape (H,). Sums (times W) to 4 pi."""
    edges = np.arange(H + 1) * np.pi / H
    return (np.cos(edges[:-1]) - np.cos(edges[1:])) * (2 * np.pi / W)


def latlong_angles(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(theta, phi) of non-zero points Q (k,3); does not require |q| = 1 and never returns NaN.

    theta = atan2(hypot(x, y), z) in [0, pi] (no arccos, so z = 1 + 1e-12 is fine);
    phi = atan2(y, x) in [-pi, pi].
    """
    Q = np.asarray(Q, dtype=np.float64)
    if not np.all(np.isfinite(Q)):
        raise ValueError("non-finite query point")
    theta = np.arctan2(np.hypot(Q[:, 0], Q[:, 1]), Q[:, 2])
    phi = np.arctan2(Q[:, 1], Q[:, 0])
    return theta, phi


def latlong_cell_index(Q: np.ndarray, H: int, W: int) -> np.ndarray:
    """Flat cell index (k,) int64 of points Q (k,3).

    Ownership rule: cells are half-open [theta_i, theta_{i+1}) x [phi_j, phi_{j+1}); theta = pi goes to
    row H-1; phi = +pi is the same meridian as -pi and goes to column 0. Raises on NaN/inf.
    """
    theta, phi = latlong_angles(Q)
    i = np.clip(np.floor(theta * (H / np.pi)).astype(np.int64), 0, H - 1)
    j = np.floor((phi + np.pi) * (W / (2 * np.pi))).astype(np.int64) % W
    return i * W + j


# ----------------------------------------------------------------------------------------- densities


def splat_kernel(t: np.ndarray, t0: float) -> np.ndarray:
    """Normalised compact biweight kernel on S^2 as a function of t = 1 - <c, p>.

    k(t) = 3 / (2 pi t0) * (1 - t/t0)^2 for t < t0, else 0. Exactly integrates to 1 over the sphere
    (int_S2 k = 2 pi int_0^t0 k dt = 1). Support is the cap of angular radius alpha0, t0 = 1 - cos alpha0.
    """
    k = 3.0 / (2 * np.pi * t0) * np.square(1.0 - t / t0)
    return np.where(t < t0, k, 0.0)


def splat_density_at(P: np.ndarray, mass: np.ndarray, Q: np.ndarray, alpha0: float,
                     tree_P: cKDTree | None = None, query_weights: np.ndarray | None = None) -> np.ndarray:
    """Kernel density sum_v mass_v k(1 - <q, p_v>) at unit points Q (k,3); mass (n,) or (n,c).

    The value depends only on dot products, so evaluating at R^T c on the original P is identical (to
    rounding) to evaluating at c on the rotated R P: the channel is exactly rotation-equivariant as a
    function; only the grid sampling and the quadrature of its total are discrete.
    """
    t0 = 1.0 - np.cos(alpha0)
    r_chord = np.sqrt(2.0 * t0)
    tree_P = cKDTree(P) if tree_P is None else tree_P
    S = cKDTree(Q).sparse_distance_matrix(tree_P, r_chord, output_type="coo_matrix")
    k = splat_kernel(0.5 * np.square(S.data), t0)
    if query_weights is not None:
        # Per-vertex discrete normalisation: divide each vertex's kernel by its own quadrature sum
        # sum_q w_q k(q, p_v), so sum_q w_q density_q = sum_v mass_v exactly. Without it the sampled
        # kernel integrates to 0.985-1.09 (+7.7% near the poles), which sum|K| ~ 1000 on CAD creases
        # turns into a signed mass error of ~2. Every vertex has quadrature mass > 0
        # because the support (alpha0 >= 1 cell height) always covers a query point.
        s_v = np.bincount(S.col, weights=k * np.asarray(query_weights, np.float64)[S.row], minlength=len(P))
        if np.any(s_v[np.unique(S.col)] <= 0) or len(np.unique(S.col)) < len(P):
            raise ValueError("splat support of some vertex covers no query point")
        k = k / s_v[S.col]
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim == 1:
        return np.bincount(S.row, weights=mass[S.col] * k, minlength=len(Q))
    return np.stack([np.bincount(S.row, weights=mass[S.col, c] * k, minlength=len(Q))
                     for c in range(mass.shape[1])], -1)


# ----------------------------------------------------------------------------------------- location


def radial_barycentrics(Q: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Barycentrics of Q radially projected onto the chordal plane of triangle (A,B,C), row-wise.

    b_A = det[q,B,C]/s, b_B = det[q,C,A]/s, b_C = det[q,A,B]/s with s the sum. This is the unique rule with
    normalize(b_A A + b_B B + b_C C) = q. Returns (b (k,3), s (k,)); s > 0 iff q is on the front side
    (q . triangle normal > 0). q lies in the closed spherical triangle iff s > 0 and min(b) >= 0.
    """
    dA = np.einsum("ij,ij->i", Q, np.cross(B, C))
    dB = np.einsum("ij,ij->i", Q, np.cross(C, A))
    dC = np.einsum("ij,ij->i", Q, np.cross(A, B))
    s = dA + dB + dC
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.stack([dA, dB, dC], -1) / s[:, None]
    return b, s


def _angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), np.einsum("ij,ij->i", u, v))


class SphereLocator:
    """Exact point location on a spherical triangulation (P unit, F CCW from outside, all det > 0).

    Build once per shape (rotation-independent). Each face is registered in every lat-long bucket cell
    that its bounding cap overlaps (cap centre = normalised vertex sum, radius = max angle to a vertex;
    a spherical triangle lies inside that cap because the cap is spherically convex). A query's
    containing face is therefore always among the candidates of the query's own bucket, whatever the
    grading of the mesh. Queries found in no candidate (only possible through rounding on edges) fall
    back to a brute-force search over all faces, choosing the face with the largest min barycentric.
    ``n_unlocated`` counts queries that even the fallback could not place in a face (invalid mesh).
    """

    def __init__(self, P: np.ndarray, F: np.ndarray, H: int = 64, W: int = 128, tol: float = 1e-12):
        self.P, self.F, self.H, self.W, self.tol = P, F, H, W, tol
        A, B, C = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        m = A + B + C
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        r = np.max(np.stack([_angle(m, A), _angle(m, B), _angle(m, C)], -1), -1) + 1e-9
        th, ph = latlong_angles(m)
        dth, dph = np.pi / H, 2 * np.pi / W
        i0 = np.clip(np.floor((th - r) / dth), 0, H - 1).astype(np.int64)
        i1 = np.clip(np.floor((th + r) / dth), 0, H - 1).astype(np.int64)
        pole = (th - r <= 0) | (th + r >= np.pi)
        with np.errstate(invalid="ignore"):
            half = np.arcsin(np.minimum(1.0, np.sin(r) / np.maximum(np.sin(th), 1e-300)))
        allcols = pole | (half >= np.pi / 2 - 1e-12) | (np.sin(r) >= np.sin(th))
        j0 = np.floor((ph - half + np.pi) / dph).astype(np.int64)
        j1 = np.floor((ph + half + np.pi) / dph).astype(np.int64)
        j0 = np.where(allcols, 0, j0)
        j1 = np.where(allcols, W - 1, j1)
        nr, nc = i1 - i0 + 1, j1 - j0 + 1
        cnt = nr * nc
        face = np.repeat(np.arange(len(F)), cnt)
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        k = np.arange(cnt.sum()) - start
        ii = i0[face] + k // nc[face]
        jj = (j0[face] + k % nc[face]) % W
        cell = ii * W + jj
        order = np.argsort(cell, kind="stable")
        self.bucket_faces = face[order]
        self.bucket_ptr = np.concatenate([[0], np.cumsum(np.bincount(cell, minlength=H * W))])
        self.n_registrations = len(face)
        self.n_fallback = 0
        self.n_unlocated = 0

    def locate(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (face (k,), bary (k,3)) for unit queries Q (k,3). Every query gets a face."""
        P, F, tol = self.P, self.F, self.tol
        cell = latlong_cell_index(Q, self.H, self.W)
        lo, hi = self.bucket_ptr[cell], self.bucket_ptr[cell + 1]
        n = hi - lo
        q_of = np.repeat(np.arange(len(Q)), n)
        pos = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
        cand = self.bucket_faces[pos]
        b, s = radial_barycentrics(Q[q_of], P[F[cand, 0]], P[F[cand, 1]], P[F[cand, 2]])
        score = np.where(s > 0, np.min(b, axis=1), -np.inf)
        face = np.full(len(Q), -1, np.int64)
        best = np.full(len(Q), -np.inf)
        # per query keep the candidate with max min-barycentric (deterministic edge ownership)
        order = np.lexsort((-score, q_of))
        first = np.ones(len(order), bool)
        first[1:] = q_of[order][1:] != q_of[order][:-1]
        sel = order[first]
        face[q_of[sel]] = cand[sel]
        best[q_of[sel]] = score[sel]
        miss = np.flatnonzero(best < -tol)
        self.n_fallback += len(miss)
        if len(miss):
            face[miss] = brute_force_locate(P, F, Q[miss])
        bary, s = radial_barycentrics(Q, P[F[face, 0]], P[F[face, 1]], P[F[face, 2]])
        with np.errstate(invalid="ignore"):
            contained = (s > 0) & (np.min(bary, axis=1) >= -tol)
        self.n_unlocated += int(np.sum(~contained))
        return face, bary


def brute_force_locate(P: np.ndarray, F: np.ndarray, Q: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Exhaustive reference: face maximising min radial barycentric (front side only) for each query."""
    A, B, C = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    nBC, nCA, nAB = np.cross(B, C), np.cross(C, A), np.cross(A, B)
    out = np.empty(len(Q), np.int64)
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        d = np.stack([q @ nBC.T, q @ nCA.T, q @ nAB.T], -1)  # (c, m, 3)
        tot = d.sum(-1)
        sc = np.where(tot > 0, d.min(-1) / np.where(tot > 0, tot, 1.0), -np.inf)
        out[s:s + chunk] = np.argmax(sc, axis=1)
    return out


def interpolate_vertex_field(locator: SphereLocator, Q: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Piecewise-linear (radial-chordal barycentric) interpolation of per-vertex values (n,) or (n,c)."""
    face, bary = locator.locate(Q)
    vals = values[locator.F[face]]  # (k,3) or (k,3,c)
    return np.einsum("kj,kj...->k...", bary, vals)


# ----------------------------------------------------------------------------------------- pooling


def cell_min(P: np.ndarray, values: np.ndarray, H: int, W: int) -> np.ndarray:
    """Per-cell minimum of a vertex field over vertices with P_v in the cell, (H*W,); empty cells are +inf."""
    mn = np.full(H * W, np.inf)
    np.minimum.at(mn, latlong_cell_index(P, H, W), values)
    return mn


# ----------------------------------------------------------------------------------------- channels


def compute_grid(P: np.ndarray, F: np.ndarray, psi: np.ndarray, K_pt: np.ndarray, H_mean: np.ndarray,
                 K_int: np.ndarray, height: int, width: int, alpha_cells: float,
                 channels: tuple[str, ...] = CHANNELS, R: np.ndarray | None = None,
                 locator: SphereLocator | None = None, tree: cKDTree | None = None) -> tuple[np.ndarray, dict]:
    """Grid (C, height, width) float64 of ``channels`` in the frame of ``P @ R.T`` (R = identity if None).

    psi, K_pt, H: radial-chordal barycentric interpolation at R^T c. psi_min: min(psi[c], min psi_v over
    vertices whose rotated position lies in c). K_int_density: biweight splat of K_int at
    alpha0 = alpha_cells * pi/height, per steradian (the kernel integrates to exactly 1, so the continuous
    total is sum K_int; each vertex's kernel is normalised by its own discrete quadrature sum, so the
    grid total equals sum K_int exactly for every rotation). vertex_density: log1p of the same splat with mass 1 per vertex.
    ``locator`` / ``tree`` (built on the unrotated P) may be passed in to reuse them across rotations.

    Returns (grid, info) with info = n_unlocated, n_fallback, mass_error = |sum density*Omega - 4 pi|.
    """
    unknown = [c for c in channels if c not in CHANNELS]
    if unknown:
        raise ValueError(f"unknown grid channels: {unknown}")
    P = np.asarray(P, np.float64)
    C = latlong_cell_centers(height, width)
    Q = C if R is None else C @ R                              # rows are (R^T c)^T
    Prot = P if R is None else P @ R.T
    locator = SphereLocator(P, F) if locator is None else locator
    tree = cKDTree(P) if tree is None else tree
    nf0, nu0 = locator.n_fallback, locator.n_unlocated

    pl = interpolate_vertex_field(locator, Q, np.column_stack([psi, K_pt, H_mean]))
    omega = np.repeat(latlong_cell_solid_angle(height, width), width)
    dens = splat_density_at(P, np.column_stack([K_int, np.ones(len(P))]), Q,
                            alpha_cells * np.pi / height, tree, query_weights=omega)
    out = {
        "psi": pl[:, 0],
        "psi_min": np.minimum(pl[:, 0], cell_min(Prot, psi, height, width)),
        "K_pt": pl[:, 1],
        "H": pl[:, 2],
        "K_int_density": dens[:, 0],
        "K_int_density_asinh": np.arcsinh(dens[:, 0]),
        "vertex_density": np.log1p(dens[:, 1]),
    }
    G = np.stack([out[c].reshape(height, width) for c in channels])
    info = {"n_unlocated": locator.n_unlocated - nu0, "n_fallback": locator.n_fallback - nf0,
            "mass_error": float(abs(np.sum(dens[:, 0] * omega) - 4 * np.pi))}
    return G, info


# ----------------------------------------------------------------------------------------- rotation


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform proper rotation (det = +1) via scipy's quaternion sampler."""
    from scipy.spatial.transform import Rotation
    R = Rotation.random(random_state=rng).as_matrix()
    assert np.linalg.det(R) > 0
    return R


def _pad_latlong(G: np.ndarray) -> np.ndarray:
    """Pad (C,H,W) by one cell: columns wrap; the row beyond a pole is the same row shifted by W/2."""
    C, H, W = G.shape
    Gp = np.concatenate([G[:, :, -1:], G, G[:, :, :1]], axis=2)
    top = np.roll(Gp[:, :1], W // 2, axis=2)
    bot = np.roll(Gp[:, -1:], W // 2, axis=2)
    return np.concatenate([top, Gp, bot], axis=1)


def rotation_sample_coords(R: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractional (row, col) in the PADDED source grid of R^T c for every target cell centre c."""
    src = latlong_cell_centers(H, W) @ R  # rows are (R^T c)^T
    th, ph = latlong_angles(src)
    y = th * (H / np.pi) - 0.5 + 1.0
    x = (ph + np.pi) * (W / (2 * np.pi)) - 0.5 + 1.0
    return y, x


def rotate_grid_torch(G, R: np.ndarray, device: str = "cpu"):
    """Option (b), the interpolation alternative: bilinear resampling of a stored grid (C,H,W) to the
    rotated frame via F.grid_sample (float32; works on mps). Smooth channels only."""
    import torch
    import torch.nn.functional as Fnn
    G = np.asarray(G, dtype=np.float32)
    C, H, W = G.shape
    Gp = torch.from_numpy(_pad_latlong(G)).to(device)[None]
    y, x = rotation_sample_coords(R, H, W)
    gx = 2 * x / (W + 1) - 1
    gy = 2 * y / (H + 1) - 1
    grid = torch.from_numpy(np.stack([gx, gy], -1).astype(np.float32).reshape(1, H, W, 2)).to(device)
    return Fnn.grid_sample(Gp, grid, mode="bilinear", align_corners=True)[0]
