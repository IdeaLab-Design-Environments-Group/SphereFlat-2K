"""psi stage against exact conformal maps (analytic oracles), not only the round sphere."""
import copy
from pathlib import Path

import numpy as np
import pytest
import trimesh

from moe_data import geometry as geo
from moe_data.config import load_config
from moe_data.stages import psi as psi_stage

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return copy.deepcopy(load_config(CONFIG))


_R = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
_R *= np.sign(np.linalg.det(_R))                                  # proper rotation (det +1)


def _sphere(subdiv=5):
    m = trimesh.creation.icosphere(subdivisions=subdiv)
    V = np.asarray(m.vertices, np.float64)
    return V / np.linalg.norm(V, axis=1, keepdims=True), np.asarray(m.faces, np.int64)


def _mobius_dilation(V, lam):
    """Map unit-sphere points by z -> lam*z in stereographic coordinates from the north pole.
    Returns (P, true psi) with psi = log(lam (1+|z|^2) / (1+lam^2 |z|^2)) (length ratio, log)."""
    z = (V[:, 0] + 1j * V[:, 1]) / (1.0 - V[:, 2])
    w = lam * z
    r2 = np.abs(w) ** 2
    P = np.stack([2 * w.real / (1 + r2), 2 * w.imag / (1 + r2), (r2 - 1) / (1 + r2)], axis=1)
    return P, np.log(lam * (1 + np.abs(z) ** 2) / (1 + r2))


def test_round_sphere_offset(cfg):
    V, F = _sphere()
    V = V / np.sqrt(geo.face_areas(V, F).sum())                  # unit area
    res = psi_stage.run("s", {"V_sub": V, "F": F, "P": V / np.linalg.norm(V, axis=1, keepdims=True)}, cfg)
    assert res.ok
    np.testing.assert_allclose(res.data["psi"], 0.5 * np.log(4 * np.pi), atol=2e-3)   # chordal deficit only
    assert abs(res.meta["psi_min"]) < 1e-12 and abs(res.meta["psi_max"]) < 1e-12   # stats relative to mean


@pytest.mark.parametrize("log_lam", [1.0, 3.0])
def test_mobius_dilation_sign_and_value(cfg, log_lam):
    V, F = _sphere()
    V = V @ _R.T                                                  # generic rotation: no vertex on the pole
    P, true = _mobius_dilation(V, np.exp(log_lam))
    res = psi_stage.run("s", {"V_sub": V, "F": F, "P": P}, cfg)
    assert res.ok, res.reason
    np.testing.assert_allclose(res.data["psi_lsq"], true, atol=1e-8)   # exact on Möbius maps
    err = np.abs(res.data["psi"] - true)
    assert np.percentile(err, 99) < 0.05                               # planned estimator: close, not exact
    assert np.corrcoef(res.data["psi"], true)[0, 1] > 0.999            # sign and shape right
    assert res.meta["lsq_residual_rms"] < 1e-8


def test_sign_mutation_is_detected(cfg):
    """Swapping the roles of surface and sphere (a sign error) must break the Möbius oracle."""
    V, F = _sphere(4)
    V = V @ _R.T
    P, true = _mobius_dilation(V, np.exp(2.0))
    wrong = psi_stage.psi_barycentric(P, F, V)
    assert np.percentile(np.abs(wrong - true), 50) > 0.5


def test_nonfinite_rejected(cfg):
    V, F = _sphere(3)
    P = V.copy()
    P[F[0, 1]] = P[F[0, 0]]                                        # zero-length sphere edge
    res = psi_stage.run("s", {"V_sub": V, "F": F, "P": P}, cfg)
    assert not res.ok and res.reason.startswith("nonfinite_psi")


def test_corner_distance_and_corner_excluded_range(cfg):
    """psi is log-singular at sharp input vertices; the corners stage finds them and the psi stage
    reports per-vertex corner distance and corner-excluded statistics."""
    from moe_data.stages import corners
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    res_c = corners.run("s", {"V": np.asarray(box.vertices, float), "F": np.asarray(box.faces)}, cfg)
    assert res_c.ok and res_c.meta["n_corners"] == 8
    m = box.subdivide_to_size(0.05)
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    P = V / np.linalg.norm(V, axis=1, keepdims=True)              # radial map: not conformal, fine here
    res = psi_stage.run("s", {"V_sub": V, "F": F, "P": P, "corner_pts": res_c.data["corner_pts"]}, cfg)
    assert res.ok, res.reason
    d = res.data["corner_dist"]
    assert np.isclose(d.min(), 0.0) and np.sum(d < 1e-12) == 8
    assert 0 < res.meta["corner_area_share"] < 0.2
    assert res.meta["psi_nocorner_range"] < res.meta["psi_range"]


def test_no_corners_gives_infinite_distance(cfg):
    V, F = _sphere(3)
    res = psi_stage.run("s", {"V_sub": V, "F": F, "P": V, "corner_pts": np.zeros((0, 3))}, cfg)
    assert res.ok and np.all(np.isinf(res.data["corner_dist"])) and res.meta["corner_area_share"] == 0.0
