"""Pure, vectorised mesh geometry (float64, no I/O). See docs/CONTRACTS.md.

Conventions: V (n,3) float64, F (m,3) int64, counter-clockwise seen from outside (outward normals).
Corner angles use atan2(|a x b|, a . b), which stays accurate and finite for slivers where
arccos(a . b / |a||b|) loses all precision or returns NaN (cosine rounded past 1).
"""
import numpy as np
import scipy.sparse as sp


def _as_mesh(V, F):
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64)


def _face_cross(V, F):
    """Unnormalised face normals (v1 - v0) x (v2 - v0); length = 2 * area."""
    return np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])


def face_areas(V, F) -> np.ndarray:
    V, F = _as_mesh(V, F)
    return 0.5 * np.linalg.norm(_face_cross(V, F), axis=1)


def vertex_areas_barycentric(V, F) -> np.ndarray:
    """One third of the area of every incident face."""
    V, F = _as_mesh(V, F)
    A = face_areas(V, F) / 3.0
    return np.bincount(F.ravel(), weights=np.repeat(A, 3), minlength=len(V))


def corner_angles(V, F) -> np.ndarray:
    """(m,3) interior angle at each corner, column k = angle at vertex F[:, k]."""
    V, F = _as_mesh(V, F)
    out = np.empty(F.shape, dtype=np.float64)
    for k in range(3):
        p = V[F[:, k]]
        a = V[F[:, (k + 1) % 3]] - p
        b = V[F[:, (k + 2) % 3]] - p
        out[:, k] = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))
    return out


def angle_defect(V, F) -> np.ndarray:
    """Integrated Gaussian curvature per vertex: 2*pi - sum of incident corner angles (closed mesh)."""
    V, F = _as_mesh(V, F)
    ang = corner_angles(V, F)
    return 2.0 * np.pi - np.bincount(F.ravel(), weights=ang.ravel(), minlength=len(V))


def cotan_laplacian(V, F) -> sp.csr_matrix:
    """Cotangent Laplacian L (n,n): L_ij = (cot a_ij + cot b_ij)/2, L_ii = -sum_j L_ij.

    Negative semi-definite, same convention as igl.cotmatrix. For a smooth surface
    (L x)_i ~ -2 H_i n_i A_i with outward n and H = 1/R on a sphere.
    """
    V, F = _as_mesh(V, F)
    n = len(V)
    rows, cols, vals = [], [], []
    for k in range(3):
        # corner k is opposite the edge (F[:, k+1], F[:, k+2])
        p = V[F[:, k]]
        a = V[F[:, (k + 1) % 3]] - p
        b = V[F[:, (k + 2) % 3]] - p
        cot = np.einsum("ij,ij->i", a, b) / np.linalg.norm(np.cross(a, b), axis=1)
        i, j = F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        rows += [i, j]
        cols += [j, i]
        vals += [0.5 * cot, 0.5 * cot]
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    L = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return (L - sp.diags(np.asarray(L.sum(axis=1)).ravel())).tocsr()


def vertex_normals(V, F) -> np.ndarray:
    """Unit area-weighted vertex normals (outward for CCW-outward faces)."""
    V, F = _as_mesh(V, F)
    fn = _face_cross(V, F)
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.linalg.norm(N, axis=1, keepdims=True)


def mean_curvature(V, F) -> np.ndarray:
    """Signed mean curvature H_i = -0.5 (L x)_i . n_i / A_i (barycentric A_i).

    +1/R on a sphere of radius R with outward normals; reversing the faces flips the sign.
    """
    V, F = _as_mesh(V, F)
    LV = cotan_laplacian(V, F) @ V
    return -0.5 * np.einsum("ij,ij->i", LV, vertex_normals(V, F)) / vertex_areas_barycentric(V, F)


def euler_characteristic(V, F) -> int:
    """chi = V - E + F, counting only vertices referenced by F and unique undirected edges."""
    _, F = _as_mesh(V, F)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    n_e = len(np.unique(E, axis=0))
    return int(len(np.unique(F)) - n_e + len(F))


def spherical_triangle_orientation(P, F) -> np.ndarray:
    """det[p0, p1, p2] per face; > 0 means not flipped (for unit P, CCW outward)."""
    P, F = _as_mesh(P, F)
    return np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]], P[F[:, 2]]))


def robust_spherical_orientation(P, F) -> np.ndarray:
    """((p1-p0) x (p2-p0)) . (p0+p1+p2) per face; > 0 means not flipped.

    Equal to 3*det[p0,p1,p2] in exact arithmetic, but det cancels to rounding noise on triangles
    whose edges are ~1e-11 (compressed protrusion tips) and reports false flips; the edge-difference
    form does not (on one L/r = 20 map: 134 false flips by det, 0 by this form).
    """
    P, F = _as_mesh(P, F)
    p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p0 + p1 + p2)


def spherical_triangle_areas(P, F) -> np.ndarray:
    """Signed solid angle per face for unit P (Van Oosterom-Strackee):
    Omega = 2 atan2(det, 1 + p0.p1 + p1.p2 + p2.p0). Sums to 4*pi on a single positive cover.
    """
    P, F = _as_mesh(P, F)
    p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)  # noqa: E731
    den = 1.0 + dot(p0, p1) + dot(p1, p2) + dot(p2, p0)
    return 2.0 * np.arctan2(spherical_triangle_orientation(P, F), den)


def centering_residual(V_sub, P, F) -> np.ndarray:
    """sum_T (A_T / sum A) * normalize(p0 + p1 + p2), MoebiusRegistration's centering condition.

    A_T are the triangle areas of V_sub, the geometry actually SUBMITTED to SphereMap
    (the float32-rounded remeshed vertices), not any earlier version of the mesh.
    """
    V_sub, F = _as_mesh(V_sub, F)
    P = np.asarray(P, dtype=np.float64)
    A = face_areas(V_sub, F)
    c = P[F[:, 0]] + P[F[:, 1]] + P[F[:, 2]]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return (A / A.sum()) @ c
