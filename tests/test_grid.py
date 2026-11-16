"""Grid helpers and the grid stage.

Oracles are analytic or exhaustive and do not reuse the helper under test where that could cancel a bug.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial import ConvexHull

from moe_data import geometry as geo
from moe_data import grid as g
from moe_data.config import load_config
from moe_data.stages import grid as grid_stage

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
H, W = 128, 256
RX90 = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # +90 deg about x: e_y -> e_z, e_z -> -e_y
ABC = np.array([1.0, 0.7, 0.5])


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config(CONFIG))
    c["grid"]["psi_source"] = "psi"          # synthetic inputs carry psi only; independent of config.yaml
    return c


# ------------------------------------------------------------------ meshes and oracles (independent of grid.py)


def icosphere(sub):
    m = trimesh.creation.icosphere(subdivisions=sub)
    V = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    return V, np.asarray(m.faces, np.int64)


def hull_sphere(n, jitter=0.3, seed=0):
    """Irregular unit-sphere mesh: jittered Fibonacci points, convex hull, faces oriented outward."""
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5 + rng.uniform(-jitter, jitter, n)
    z = np.clip(1 - 2 * i / n, -1, 1)
    phi = np.pi * (1 + 5**0.5) * np.arange(n) + rng.uniform(-jitter, jitter, n) * 2 / np.sqrt(n)
    P = np.column_stack([np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    F = ConvexHull(P).simplices.astype(np.int64)
    flip = np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])) < 0
    F[flip] = F[flip][:, ::-1]
    return P, F


def ellipsoid_radius(U):
    return 1.0 / np.sqrt(np.sum(np.square(U / ABC), axis=1))


def ellipsoid_density_on_sphere(U):
    """Analytic K dA/dOmega at radial-projection direction U: K(x) r^2 / (n_hat . u)."""
    r = ellipsoid_radius(U)
    X = U * r[:, None]
    a, b, c = ABC
    K = 1.0 / (a * a * b * b * c * c * np.square(np.sum(np.square(X / ABC**2), axis=1)))
    n = X / ABC**2
    return K * r * r * np.linalg.norm(n, axis=1) / np.einsum("ij,ij->i", n, U)


def dilate(V, lam):
    """Moebius dilation fixing the poles; compresses the north side by 1/lam. Returns (P, stretch)."""
    z = V[:, 2]
    den = lam * lam * (1 + z) + (1 - z)
    s = 2 * lam / den
    P = np.column_stack([V[:, 0] * s, V[:, 1] * s, (lam * lam * (1 + z) - (1 - z)) / den])
    return P / np.linalg.norm(P, axis=1, keepdims=True), s


def gauss_legendre_sphere_integral(fun, n=400):
    x, w = np.polynomial.legendre.leggauss(n)
    phi = (np.arange(2 * n) + 0.5) * np.pi / n
    Z, PH = np.meshgrid(x, phi, indexing="ij")
    S = np.sqrt(1 - Z**2)
    U = np.stack([S * np.cos(PH), S * np.sin(PH), Z], -1).reshape(-1, 3)
    return float(np.sum(fun(U) * np.repeat(w, 2 * n) * (np.pi / n)))


@pytest.fixture(scope="module")
def sphere15k():
    return hull_sphere(15000)


@pytest.fixture(scope="module")
def ellipsoid15k():
    U, F = hull_sphere(15000, seed=1)
    return U * ellipsoid_radius(U)[:, None], F, U


def _inputs(V, F, P, psi=None):
    area = geo.vertex_areas_barycentric(V, F)
    K_int = geo.angle_defect(V, F)
    psi = np.zeros(len(V)) if psi is None else psi
    return {"V_sub": V, "F": F, "P": P, "area": area, "K_int": K_int, "K_pt": K_int / area,
            "H": geo.mean_curvature(V, F), "psi": psi, "psi_lsq": psi + 1.0}


# ------------------------------------------------------------------ cell index


@pytest.mark.parametrize("q,row,col", [
    ((-1.0, 0.0, 0.0), H // 2, 0),          # phi = +pi -> column 0 (same meridian as -pi)
    ((-1.0, -0.0, 0.0), H // 2, 0),         # phi = -pi
    ((-1.0, -1e-300, 0.0), H // 2, 0),
    ((0.0, 0.0, 1.0), 0, W // 2),           # north pole, phi = atan2(0, 0) = 0
    ((0.0, 0.0, 1.0 + 1e-12), 0, W // 2),   # slightly non-unit, no NaN
    ((0.0, 0.0, -1.0), H - 1, W // 2),
    ((0.0, 0.0, -1.0 - 1e-12), H - 1, W // 2),
    ((1.0, 0.0, 0.0), H // 2, W // 2),      # theta = pi/2 exactly belongs to the lower row (half-open)
])
def test_cell_index_edge_cases(q, row, col):
    idx = g.latlong_cell_index(np.array([q]), H, W)[0]
    assert (idx // W, idx % W) == (row, col)


def test_cell_index_rejects_nan():
    with pytest.raises(ValueError):
        g.latlong_cell_index(np.array([[np.nan, 0, 1.0]]), H, W)


def test_cell_index_against_independent_bounds():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(20000, 3))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    idx = g.latlong_cell_index(Q, H, W)
    i, j = idx // W, idx % W
    assert np.all((Q[:, 2] <= np.cos(i * np.pi / H) + 1e-15) & (Q[:, 2] >= np.cos((i + 1) * np.pi / H) - 1e-15))
    phi = np.arctan2(Q[:, 1], Q[:, 0])
    assert np.all(np.abs(np.angle(np.exp(1j * (phi - (-np.pi + (j + 0.5) * 2 * np.pi / W))))) <= np.pi / W + 1e-12)


def test_cell_centres_are_in_their_own_cell_and_solid_angle_sums_to_4pi():
    C = g.latlong_cell_centers(H, W)
    assert np.array_equal(g.latlong_cell_index(C, H, W), np.arange(H * W))
    assert np.isclose(g.latlong_cell_solid_angle(H, W).sum() * W, 4 * np.pi, rtol=1e-13)


# ------------------------------------------------------------------ barycentrics and location


def test_radial_barycentrics_recover_weights():
    rng = np.random.default_rng(1)
    A, B, C = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in rng.normal(size=(3, 500, 3)))
    flip = np.einsum("ij,ij->i", A, np.cross(B, C)) < 0
    B[flip], C[flip] = C[flip].copy(), B[flip].copy()
    w = rng.dirichlet([1, 1, 1], size=500)
    q = w[:, :1] * A + w[:, 1:2] * B + w[:, 2:] * C
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    b, s = g.radial_barycentrics(q, A, B, C)
    assert np.all(s > 0)
    np.testing.assert_allclose(b, w, atol=1e-9)


def test_locator_matches_exhaustive_containment_on_graded_mesh():
    V, F = icosphere(5)
    P, _ = dilate(V, np.e**3)
    rng = np.random.default_rng(2)
    C = g.latlong_cell_centers(H, W)
    edge_mid = P[F[:, 0]] + P[F[:, 1]]
    edge_mid /= np.linalg.norm(edge_mid, axis=1, keepdims=True)
    t = np.linspace(-1.5, 1.5, 200)
    seam = np.column_stack([-np.cos(t), np.full(200, 1e-17), np.sin(t)])
    Q = np.concatenate([C, C @ g.random_rotation(rng), P, edge_mid[::7], seam, [[0, 0, 1.0], [0, 0, -1.0]]])
    loc = g.SphereLocator(P, F)
    face, bary = loc.locate(Q)
    A, B, Cc = P[F[face, 0]], P[F[face, 1]], P[F[face, 2]]
    d = np.stack([np.einsum("ij,ij->i", Q, np.cross(B, Cc)), np.einsum("ij,ij->i", Q, np.cross(Cc, A)),
                  np.einsum("ij,ij->i", Q, np.cross(A, B))], -1)
    assert np.all(d.min(1) >= -1e-10 * np.abs(d).sum(1)), "located face does not contain query"
    assert loc.n_fallback == 0 and loc.n_unlocated == 0
    sub = np.random.default_rng(3).choice(len(Q), 3000, replace=False)
    bf = g.brute_force_locate(P, F, Q[sub])
    interior = bary[sub].min(1) > 1e-9
    assert np.array_equal(bf[interior], face[sub][interior])


def test_locator_counts_unlocated_on_flipped_face():
    V, F = icosphere(1)
    F = F.copy()
    F[0] = F[0, ::-1]                       # det < 0: its interior is covered by no front-facing face
    loc = g.SphereLocator(V, F)
    loc.locate(g.latlong_cell_centers(64, 128))
    assert loc.n_unlocated > 0


# ------------------------------------------------------------------ rotation oracles


def test_vertex_rotation_oracle_f_equals_z():
    """Rotate per-vertex positions literally, resample: f = z must become -y, x*y must become x*z."""
    V, F = icosphere(5)
    loc = g.SphereLocator(V @ RX90.T, F)
    C = g.latlong_cell_centers(H, W)
    np.testing.assert_allclose(g.interpolate_vertex_field(loc, C, V[:, 2]), -C[:, 1], atol=5e-4)
    np.testing.assert_allclose(g.interpolate_vertex_field(loc, C, V[:, 0] * V[:, 1]), C[:, 0] * C[:, 2], atol=5e-4)


def test_compute_grid_with_R_equals_grid_of_rotated_mesh():
    """Exact option (a): evaluating at R^T c on P equals evaluating at c on P @ R.T, every channel."""
    V, F = icosphere(4)
    P, s = dilate(V, np.e)
    rng = np.random.default_rng(8)
    R = g.random_rotation(rng)
    f = [np.log(s), V[:, 0], V[:, 1] * V[:, 2], np.abs(rng.normal(size=len(V))) * 1e-3]
    a, _ = g.compute_grid(P, F, *f, 32, 64, 2.0, R=R)
    b, _ = g.compute_grid(P @ R.T, F, *f, 32, 64, 2.0)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_grid_rotation_torch_oracle_and_linear_fields_through_poles():
    C = g.latlong_cell_centers(H, W)
    G = np.stack([C[:, 2], C[:, 0] * C[:, 1]]).reshape(2, H, W)
    out = g.rotate_grid_torch(G, RX90).numpy()
    np.testing.assert_allclose(out[0].ravel(), -C[:, 1], atol=2e-3)
    np.testing.assert_allclose(out[1].ravel(), C[:, 0] * C[:, 2], atol=2e-3)
    a = 2 * 2 * np.pi / W                   # rotation about z by two columns is a pure shift
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Gr = np.random.default_rng(4).normal(size=(1, H, W))
    np.testing.assert_allclose(g.rotate_grid_torch(Gr, Rz).numpy()[0], np.roll(Gr[0], 2, axis=1), atol=2e-4)
    rng = np.random.default_rng(7)
    for _ in range(3):
        R = g.random_rotation(rng)
        np.testing.assert_allclose(g.rotate_grid_torch(C.T.reshape(3, H, W), R).numpy().reshape(3, -1),
                                   (C @ R).T, atol=1e-3)


def test_random_rotation_is_proper():
    rng = np.random.default_rng(5)
    Rs = np.array([g.random_rotation(rng) for _ in range(2000)])
    assert np.all(np.linalg.det(Rs) > 0.999999)
    np.testing.assert_allclose(np.einsum("kij,klj->kil", Rs, Rs), np.broadcast_to(np.eye(3), Rs.shape), atol=1e-12)
    assert np.abs(Rs.mean(0)).max() < 0.05


# ------------------------------------------------------------------ splat kernel


@pytest.mark.parametrize("alpha0", [0.02, 0.05, 0.1, 0.5])
def test_splat_kernel_integrates_to_one(alpha0):
    t0 = 1 - np.cos(alpha0)
    assert abs(gauss_legendre_sphere_integral(lambda U: g.splat_kernel(1 - U[:, 2], t0), n=2000) - 1) < 1e-4
    mu = np.array([0.3, -0.5, 0.81])
    mu /= np.linalg.norm(mu)
    val = gauss_legendre_sphere_integral(lambda U: g.splat_kernel(1 - U @ mu, t0), n=2000)
    assert abs(val - 1) < (2e-2 if alpha0 < 0.03 else 1e-3)


def test_splat_density_is_rotation_equivariant_pointwise():
    V, F = icosphere(4)
    m = geo.angle_defect(V, F)
    R = g.random_rotation(np.random.default_rng(6))
    Q = g.latlong_cell_centers(32, 64)
    np.testing.assert_allclose(g.splat_density_at(V @ R.T, m, Q, 0.1), g.splat_density_at(V, m, Q @ R, 0.1),
                               atol=1e-9)


# ------------------------------------------------------------------ grid stage


def _channel(res, name):
    return res.data["grid"][list(res.data["grid_channels"]).index(name)].astype(np.float64)


def test_stage_on_unit_sphere(cfg, sphere15k):
    V, F = sphere15k
    res = grid_stage.run("s", _inputs(V, F, V), cfg)
    assert res.ok, res.reason
    assert res.data["grid"].dtype == np.float32 and res.data["grid"].shape == (7, H, W)
    assert list(res.data["grid_channels"]) == cfg["grid"]["channels"]
    K = _channel(res, "K_int_density")
    assert np.sqrt(np.mean((K - 1) ** 2)) < 0.05                  # truth is 1 in every cell
    np.testing.assert_allclose(_channel(res, "K_int_density_asinh"), np.arcsinh(K), atol=1e-6)
    assert res.meta["mass_error"] < 2e-4 * 4 * np.pi and res.meta["n_unlocated"] == 0
    vd = np.expm1(_channel(res, "vertex_density"))                # count density ~ n / 4pi
    assert abs(np.median(vd) / (len(V) / (4 * np.pi)) - 1) < 0.02
    Hg, Hv = _channel(res, "H"), geo.mean_curvature(V, F)           # discrete H is noisy pointwise; PL
    assert Hv.min() - 1e-6 <= Hg.min() and Hg.max() <= Hv.max() + 1e-6  # interpolation stays in range
    assert abs(np.median(Hg) - 1) < 0.01
    np.testing.assert_allclose(_channel(res, "psi"), 0.0, atol=0)


def test_stage_ellipsoid_density_and_moments(cfg, ellipsoid15k):
    V, F, U = ellipsoid15k
    inp = _inputs(V, F, U)
    res = grid_stage.run("e", inp, cfg)
    assert res.ok, res.reason
    G = _channel(res, "K_int_density").ravel()
    C = g.latlong_cell_centers(H, W)
    rho = ellipsoid_density_on_sphere(C)
    assert np.sqrt(np.mean((G - rho) ** 2)) / np.sqrt(np.mean(rho**2)) < 0.03
    omega = np.repeat(g.latlong_cell_solid_angle(H, W), W)
    K_int = inp["K_int"]
    for h in (lambda X: np.ones(len(X)), lambda X: X[:, 2], lambda X: X[:, 0] ** 2,
              lambda X: X[:, 0] * X[:, 1] + 0.5 * X[:, 0], lambda X: np.exp(2 * X[:, 1]),
              lambda X: 3 * X[:, 2] ** 2 - 1):
        err = abs(np.sum(G * h(C) * omega) - np.sum(K_int * h(U))) / np.sum(np.abs(K_int * h(U)))
        assert err < 5e-3


def test_stage_psi_min_keeps_vertex_minimum_and_psi_source(cfg, sphere15k):
    V, F = sphere15k
    P, s = dilate(V, np.e**3)
    psi = np.log(s)
    inp = {**_inputs(V, F, P, psi), "psi_lsq": psi - 0.25}
    res = grid_stage.run("d", inp, cfg)
    assert res.ok, res.reason
    assert _channel(res, "psi_min").min() == pytest.approx(psi.min(), abs=1e-6)  # float32 storage
    assert np.all(_channel(res, "psi_min") <= _channel(res, "psi"))
    c2 = copy.deepcopy(cfg)
    c2["grid"]["psi_source"] = "psi_lsq"
    res2 = grid_stage.run("d", inp, c2)
    np.testing.assert_allclose(_channel(res2, "psi"), _channel(res, "psi") - 0.25, atol=1e-5)


def test_stage_rejects_unlocatable_and_nonfinite_and_missing(cfg):
    V, F = icosphere(2)
    Fb = F.copy()
    Fb[0] = Fb[0, ::-1]
    res = grid_stage.run("f", _inputs(V, Fb, V), cfg)
    assert not res.ok and res.reason.startswith("locate_failed")
    inp = _inputs(V, F, V)
    inp["K_pt"] = inp["K_pt"].copy()
    inp["K_pt"][3] = np.nan
    res = grid_stage.run("n", inp, cfg)
    assert not res.ok and "nonfinite_grid" in res.reason and "K_pt" in res.reason
    inp.pop("psi")
    res = grid_stage.run("m", inp, cfg)
    assert not res.ok and res.reason == "missing_input: psi"


def test_splat_mass_exact_with_cancelling_masses_and_poles():
    """Per-vertex discrete kernel sums vary by ~1% (+7.7% at the poles); with
    sum|K| >> 4 pi (CAD creases) the signed mass error reached 2.08. Per-vertex discrete normalisation
    makes sum density * Omega equal sum mass exactly, for any rotation."""
    from moe_data import grid as g
    rng = np.random.default_rng(3)
    P = rng.normal(size=(4000, 3))
    P[:20] = [0, 0, 1]                                     # vertices exactly at and near the poles
    P[20:40] = [0, 0.002, -1]
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    mass = rng.normal(scale=0.25, size=len(P))             # sum|m| ~ 800, sum m ~ 0
    H, W = 64, 128
    omega = np.repeat(g.latlong_cell_solid_angle(H, W), W)
    for R in [np.eye(3), g.random_rotation(rng)]:
        Q = g.latlong_cell_centers(H, W) @ R
        dens = g.splat_density_at(P, mass, Q, 2.0 * np.pi / H, query_weights=omega)
        assert abs(np.sum(dens * omega) - mass.sum()) < 1e-9
