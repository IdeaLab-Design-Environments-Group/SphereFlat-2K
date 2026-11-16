import igl
import numpy as np
import trimesh
from numpy.testing import assert_allclose

from moe_data import geometry as geo

ABC = np.array([1.0, 0.7, 0.5])


def _icosphere(subdivisions=3, radius=1.0):
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def _ellipsoid(subdivisions):
    V, F = _icosphere(subdivisions)
    return V * ABC, F


def _ellipsoid_K(V):
    a, b, c = ABC
    x, y, z = V.T
    return 1.0 / (a * a * b * b * c * c * (x**2 / a**4 + y**2 / b**4 + z**2 / c**4) ** 2)


def _ellipsoid_H(V):
    a, b, c = ABC
    x, y, z = V.T
    s = x**2 / a**4 + y**2 / b**4 + z**2 / c**4
    return np.abs(x * x + y * y + z * z - a * a - b * b - c * c) / (2 * (a * b * c) ** 2 * s**1.5)


def _torus(nu=40, nv=20, R=1.0, r=0.3):
    u, v = np.meshgrid(np.arange(nu) * 2 * np.pi / nu, np.arange(nv) * 2 * np.pi / nv, indexing="ij")
    V = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], -1)
    idx = lambda i, j: (i % nu) * nv + (j % nv)  # noqa: E731
    F = []
    for i in range(nu):
        for j in range(nv):
            F += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    return V.reshape(-1, 3), np.array(F, dtype=np.int64)


# ---------------------------------------------------------------- areas

def test_face_and_vertex_areas_right_triangle():
    V = np.array([[0.0, 0, 0], [3, 0, 0], [0, 4, 0]])
    F = np.array([[0, 1, 2]])
    assert_allclose(geo.face_areas(V, F), [6.0], rtol=1e-15)
    assert_allclose(geo.vertex_areas_barycentric(V, F), [2.0, 2.0, 2.0], rtol=1e-15)


def test_areas_on_unit_area_icosphere():
    V, F = _icosphere(3)
    V = V / np.sqrt(geo.face_areas(V, F).sum())
    A = geo.face_areas(V, F)
    assert_allclose(A.sum(), 1.0, rtol=1e-12)
    assert_allclose(A, 0.5 * igl.doublearea(V, F), rtol=1e-12)  # independent implementation
    Av = geo.vertex_areas_barycentric(V, F)
    assert Av.shape == (len(V),)
    assert_allclose(Av.sum(), 1.0, rtol=1e-12)
    assert np.all(Av > 0)


def test_unreferenced_vertex_gets_zero_area():
    V = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]])
    F = np.array([[0, 1, 2]])
    assert geo.vertex_areas_barycentric(V, F)[3] == 0.0


# ---------------------------------------------------------------- angles

def test_corner_angles_sum_to_pi_and_match_known():
    V = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    ang = geo.corner_angles(V, np.array([[0, 1, 2]]))
    assert_allclose(ang[0], [np.pi / 2, np.pi / 4, np.pi / 4], rtol=1e-14)


def test_atan2_angles_finite_where_arccos_is_nan():
    # Mutation check: with the arccos formula this collinear corner gives cos = 1 + 2^-52 -> NaN.
    a = np.array([-1.303, 0.905, 0.446])
    V = np.array([[0.0, 0, 0], a, 2 * a])
    with np.errstate(invalid="ignore"):
        arccos_angle = np.arccos(a @ (2 * a) / (np.linalg.norm(a) * np.linalg.norm(2 * a)))
    assert np.isnan(arccos_angle)
    ang = geo.corner_angles(V, np.array([[0, 1, 2]]))
    assert np.all(np.isfinite(ang))
    assert_allclose(ang[0, 0], 0.0, atol=1e-15)


def test_sliver_angle_accurate_where_arccos_loses_it():
    # Mutation check: arccos gives exactly 0 for a 1e-9 rad corner; atan2 recovers it.
    eps = 1e-9
    V = np.array([[0.0, 0, 0], [1, 0, 0], [1, eps, 0]])
    u, w = V[1], V[2]
    assert np.arccos(u @ w / (np.linalg.norm(u) * np.linalg.norm(w))) == 0.0
    assert_allclose(geo.corner_angles(V, np.array([[0, 1, 2]]))[0, 0], np.arctan(eps), rtol=1e-12)


# ---------------------------------------------------------------- Gaussian curvature

def test_ellipsoid_gaussian_curvature_converges():
    errs = []
    for s in (3, 4):
        V, F = _ellipsoid(s)
        K = geo.angle_defect(V, F) / geo.vertex_areas_barycentric(V, F)
        rel = np.abs(K / _ellipsoid_K(V) - 1)
        errs.append(np.mean(rel))
        # only the 12 valence-5 vertices of the icosphere carry a resolution-independent bias
        regular = np.bincount(F.ravel(), minlength=len(V)) == 6
        assert np.sum(~regular) == 12
        assert np.max(rel[regular]) < 0.05
        assert np.max(rel) < 0.2
    assert errs[1] < 0.004
    assert errs[0] / errs[1] > 2.5  # first-order-in-area convergence (observed ~3.4)


def test_gauss_bonnet_noisy_ellipsoid():
    V, F = _ellipsoid(3)
    rng = np.random.default_rng(0)
    V = V * (1 + 0.05 * rng.uniform(-1, 1, size=(len(V), 1)))
    assert geo.euler_characteristic(V, F) == 2
    assert_allclose(geo.angle_defect(V, F).sum(), 4 * np.pi, atol=1e-10)


def test_gauss_bonnet_torus_is_zero():
    V, F = _torus()
    assert geo.euler_characteristic(V, F) == 0
    assert_allclose(geo.angle_defect(V, F).sum(), 0.0, atol=1e-10)


def test_euler_characteristic_ignores_unreferenced_vertices():
    V, F = _icosphere(1)
    V = np.vstack([V, [[9.0, 9, 9]]])
    assert geo.euler_characteristic(V, F) == 2


# ---------------------------------------------------------------- mean curvature

def test_cotan_laplacian_matches_igl():
    V, F = _ellipsoid(2)
    L = geo.cotan_laplacian(V, F)
    assert abs(L - igl.cotmatrix(V, F)).max() < 1e-12
    assert_allclose(L @ np.ones(len(V)), 0.0, atol=1e-12)


def test_mean_curvature_sphere_radius_2_outward_positive():
    V, F = _icosphere(4, radius=2.0)
    H = geo.mean_curvature(V, F)
    assert_allclose(np.median(H), 0.5, rtol=0.01)
    assert np.all(np.abs(H - 0.5) < 0.08)  # worst case at the 12 valence-5 vertices
    # Mutation check: a negated sign convention fails the oracle.
    assert not np.allclose(np.median(-H), 0.5, rtol=0.01)


def test_mean_curvature_sign_flips_with_face_orientation():
    V, F = _ellipsoid(3)
    H = geo.mean_curvature(V, F)
    assert_allclose(geo.mean_curvature(V, F[:, ::-1]), -H, rtol=1e-12, atol=1e-12)
    assert np.all(H > 0)


def test_ellipsoid_mean_curvature_matches_analytic():
    V, F = _ellipsoid(4)
    rel = np.abs(geo.mean_curvature(V, F) / _ellipsoid_H(V) - 1)
    assert np.mean(rel) < 0.005
    assert np.quantile(rel, 0.99) < 0.05


# ---------------------------------------------------------------- spherical triangles

def test_spherical_areas_sum_to_4pi_on_icosphere():
    V, F = _icosphere(3)
    P = V / np.linalg.norm(V, axis=1, keepdims=True)
    assert np.all(geo.spherical_triangle_orientation(P, F) > 0)
    Om = geo.spherical_triangle_areas(P, F)
    assert np.all(Om > 0)
    assert_allclose(Om.sum(), 4 * np.pi, rtol=1e-13)


def test_octant_triangle_area_and_sign():
    P = np.eye(3)
    F = np.array([[0, 1, 2]])
    assert_allclose(geo.spherical_triangle_orientation(P, F), [1.0], rtol=1e-15)
    assert_allclose(geo.spherical_triangle_areas(P, F), [np.pi / 2], rtol=1e-15)
    assert_allclose(geo.spherical_triangle_areas(P, F[:, ::-1]), [-np.pi / 2], rtol=1e-15)


def test_small_spherical_triangle_area_matches_planar_limit():
    e = 1e-4
    P = np.array([[0.0, 0, 1], [e, 0, 1], [0, e, 1]])
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    F = np.array([[0, 1, 2]])
    assert_allclose(geo.spherical_triangle_areas(P, F), geo.face_areas(P, F), rtol=1e-6)


# ---------------------------------------------------------------- centering residual

def test_centering_residual_zero_for_symmetric_map_and_nonzero_when_shifted():
    V, F = _icosphere(3)
    P = V / np.linalg.norm(V, axis=1, keepdims=True)
    assert np.linalg.norm(geo.centering_residual(V, P, F)) < 1e-12
    Q = P + [0, 0, 0.3]
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    r = geo.centering_residual(V, Q, F)
    assert r[2] > 0.05 and abs(r[0]) < 1e-12 and abs(r[1]) < 1e-12


def test_centering_residual_weights_by_submitted_areas():
    # Same P, different submitted geometry: weights come from V_sub, so the residual changes.
    V, F = _icosphere(2)
    P = V / np.linalg.norm(V, axis=1, keepdims=True)
    V_heavy_top = V * np.where(V[:, 2:] > 0, 2.0, 1.0)
    assert geo.centering_residual(V_heavy_top, P, F)[2] > 0.05
