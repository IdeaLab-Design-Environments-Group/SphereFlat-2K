"""Closed-form cone scale factor u vs the mesh Poisson solve L u = K_int - Theta (moe_data.stretch)."""
import numpy as np
import pytest
import trimesh

from moe_data import geometry as geo
from moe_data import stretch as st

_R = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
_R *= np.sign(np.linalg.det(_R))                                  # proper rotation (det +1)
_AXES = np.vstack([np.eye(3), -np.eye(3)])
_THETA6 = np.full(6, 2 * np.pi / 3)


def _octasphere(n_subdiv):
    """Unit sphere from a subdivided octahedron: the 6 axis vertices are equivalent (valence 4)."""
    V = _AXES[[0, 3, 1, 4, 2, 5]].astype(float)                  # +x,-x,+y,-y,+z,-z
    F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    for _ in range(n_subdiv):
        V, F = trimesh.remesh.subdivide(V, F)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return V, np.asarray(F, np.int64)


def _axis_cones(V, R=np.eye(3)):
    return np.array([int(np.argmax(V @ (R @ a))) for a in _AXES])


def _mobius_dilation(V, lam):
    """Same oracle as tests/test_psi.py: z -> lam*z in stereographic coordinates; returns (P, psi)."""
    z = (V[:, 0] + 1j * V[:, 1]) / (1.0 - V[:, 2])
    w = lam * z
    r2 = np.abs(w) ** 2
    P = np.stack([2 * w.real / (1 + r2), 2 * w.imag / (1 + r2), (r2 - 1) / (1 + r2)], axis=1)
    return P, np.log(lam * (1 + np.abs(z) ** 2) / (1 + r2))


def _round_unit_area(n):
    V, F = _octasphere(n)
    V = V / np.sqrt(geo.face_areas(V, F).sum())
    P = V / np.linalg.norm(V, axis=1, keepdims=True)
    return V, F, P, np.full(len(V), 0.5 * np.log(4 * np.pi))


def test_laplacian_sign_on_known_function():
    """Delta z = -2 z on the unit sphere, so (L z)_i ~ -2 z_i A_i (sign convention the Poisson form uses)."""
    V, F = _octasphere(5)
    Lz = geo.cotan_laplacian(V, F) @ V[:, 2]
    A = geo.vertex_areas_barycentric(V, F)
    sel = np.abs(V[:, 2]) > 0.3
    ratio = Lz[sel] / (-2 * V[sel, 2] * A[sel])
    assert abs(np.median(ratio) - 1.0) < 0.01                    # barycentric A is off at the valence-4 poles
    assert np.mean(np.abs(ratio - 1.0) < 0.02) > 0.99


def test_closed_form_nan_at_cones_only():
    V, F, P, psi = _round_unit_area(3)
    cones = _axis_cones(V)
    u = st.closed_form_u(P, psi, cones, _THETA6)
    assert np.all(np.isnan(u[cones])) and np.isfinite(np.delete(u, cones)).all()


def test_poisson_solution_satisfies_equation_and_zero_mean():
    V, F, _, _ = _round_unit_area(3)
    cones = _axis_cones(V)
    u = st.poisson_u(V, F, cones, _THETA6)
    b = geo.angle_defect(V, F)
    b[cones] -= _THETA6
    np.testing.assert_allclose(geo.cotan_laplacian(V, F) @ u, b, atol=1e-9)
    np.testing.assert_allclose(np.sum(geo.vertex_areas_barycentric(V, F) * u), 0.0, atol=1e-12)


def test_round_sphere_agreement_converges():
    errs = []
    for n in (4, 5):
        V, F, P, psi = _round_unit_area(n)
        r = st.compare(V, F, P, psi, np.full(len(V), np.inf), _axis_cones(V), _THETA6, exclude_radius=0.04)
        assert r["masked_rms"] < 0.01 * r["u_range"]
        errs.append(r["masked_max"])
    assert errs[1] < 5e-3
    assert errs[0] / errs[1] > 1.5


@pytest.mark.parametrize("log_lam", [0.5, 1.0])
def test_mobius_dilated_sphere_agreement(log_lam):
    """Surface = round sphere, sphere map = Möbius dilation (P != V), analytic psi."""
    V, F = _octasphere(5)
    V = V @ _R.T                                                  # no vertex on the stereographic pole
    P, psi = _mobius_dilation(V, np.exp(log_lam))
    cones = _axis_cones(V, _R)
    r = st.compare(V, F, P, psi, np.full(len(V), np.inf), cones, _THETA6, exclude_radius=0.15)
    assert np.ptp(psi) > 0.9                                      # non-trivial sphere map
    # u depends only on M and the cone positions on M: the Möbius dilation must leave it unchanged
    V0, F0, P0, psi0 = V, F, V, np.zeros(len(V))
    r0 = st.compare(V0, F0, P0, psi0, np.full(len(V), np.inf), cones, _THETA6, exclude_radius=0.15)
    assert r["u_range"] == pytest.approx(r0["u_range"], abs=0.01)
    assert r["masked_max"] < 0.01 and r["masked_p99"] < 5e-3
    # the corner mask is not applied when there are no corners: both statistics coincide
    assert r["masked_max"] == pytest.approx(r["unmasked_max"])


def test_sign_mutations_are_detected():
    V, F = _octasphere(5)
    V = V @ _R.T
    P, psi = _mobius_dilation(V, np.exp(1.0))
    cones = _axis_cones(V, _R)
    area = geo.vertex_areas_barycentric(V, F)
    keep = np.min(np.linalg.norm(V[:, None] - V[cones][None], axis=2), axis=1) >= 0.15
    uc = st.closed_form_u(P, psi, cones, _THETA6)

    def err(up):
        d = (uc - up)[keep]
        return np.max(np.abs(d - np.sum(area[keep] * d) / area[keep].sum()))

    good = err(st.poisson_u(V, F, cones, _THETA6))
    flip_theta = err(st.poisson_u(V, F, cones, -_THETA6))         # Theta enters with the wrong sign
    flip_L = err(-st.poisson_u(V, F, cones, _THETA6))             # positive semi-definite L convention
    assert good < 0.01
    assert flip_theta > 50 * good and flip_L > 50 * good


def test_corner_mask_and_cone_disk_exclusion():
    V, F, P, psi = _round_unit_area(4)
    cones = _axis_cones(V)
    corner_dist = np.linalg.norm(V - V[np.argmax(V @ np.array([1.0, 1, 1]))], axis=1)  # one fake corner
    r = st.compare(V, F, P, psi, corner_dist, cones, _THETA6, exclude_radius=0.04)
    near_cone = np.min(np.linalg.norm(V[:, None] - V[cones][None], axis=2), axis=1) < 0.04
    assert r["masked_n"] == np.sum(~near_cone & (corner_dist >= 0.04))
    assert r["unmasked_n"] == np.sum(~near_cone)
    assert r["masked_n"] < r["unmasked_n"]


def test_cone_layouts():
    V, F, P, psi = _round_unit_area(4)
    fp = st.farthest_point_cones(P, 6, start=int(np.argmax(P[:, 2])))
    np.testing.assert_allclose(np.sort(np.abs(P[fp]).max(axis=1)), 1.0, atol=1e-12)   # octahedron axes
    rng = np.random.default_rng(1)
    psi2 = rng.normal(size=len(V))
    c = st.compressed_cones(V, psi2, 4, min_sep=0.2)
    assert c[0] == np.argmin(psi2) and len(set(c.tolist())) == 4
    assert min(np.linalg.norm(V[i] - V[j]) for i in c for j in c if i != j) >= 0.2
    with pytest.raises(ValueError):
        st.compressed_cones(V, psi2, 4, min_sep=10.0)
