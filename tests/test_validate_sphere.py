import copy
from pathlib import Path

import numpy as np
import pytest
import trimesh

from moe_data.config import load_config
from moe_data.stages import validate_sphere

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return copy.deepcopy(load_config(CONFIG))


def _inputs(V, F, P):
    V_sub = V.astype(np.float32).astype(np.float64)
    return {"V_sub": V_sub, "F": F.astype(np.int64), "P": P, "X": V_sub.astype(np.float32)}


@pytest.fixture
def ico():
    m = trimesh.creation.icosphere(subdivisions=3)
    V, F = np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)
    V_sub = V.astype(np.float32).astype(np.float64)
    P = V_sub / np.linalg.norm(V_sub, axis=1, keepdims=True)
    return _inputs(V_sub, F, P)


def _codes(res):
    return {part.split(":")[0] for part in res.reason.split("; ") if part}


def _bipyramid(turns):
    """Octagonal bipyramid; equator vertex k sent to longitude 2*pi*turns*k/8 on the sphere."""
    k = np.arange(8)
    ring = lambda t: np.stack([np.cos(2 * np.pi * t * k / 8), np.sin(2 * np.pi * t * k / 8), 0 * k], -1)  # noqa: E731
    poles = np.array([[0.0, 0, 1], [0, 0, -1]])
    V = np.vstack([poles, ring(1)])
    P = np.vstack([poles, ring(turns)])
    e = 2 + k
    e1 = 2 + (k + 1) % 8
    F = np.concatenate([np.stack([0 * k, e, e1], -1), np.stack([0 * k + 1, e1, e], -1)])
    return V, F, P


def test_accepts_icosphere(cfg, ico):
    res = validate_sphere.run("s", ico, cfg)
    assert res.ok, res.reason
    assert res.meta["n_flipped"] == 0 and res.meta["n_degenerate"] == 0
    assert abs(res.meta["coverage_err"]) < 1e-12
    assert res.meta["centering_residual"] < 1e-12
    assert res.meta["max_norm_err"] < 1e-15


def test_accepts_single_cover_bipyramid(cfg):
    res = validate_sphere.run("s", _inputs(*_bipyramid(1)), cfg)
    assert res.ok, res.reason


def test_rejects_double_cover_bipyramid(cfg):
    res = validate_sphere.run("s", _inputs(*_bipyramid(2)), cfg)
    assert not res.ok
    assert _codes(res) == {"coverage_not_single"}  # every other check passes
    assert res.meta["n_flipped"] == 0 and res.meta["min_orient"] > 0
    assert res.meta["centering_residual"] < cfg["validate"]["centering_tol"]
    np.testing.assert_allclose(res.meta["coverage_err"], 1.0, atol=1e-12)  # area 8*pi


def test_rejects_nan_row(cfg, ico):
    ico["P"][5] = np.nan
    res = validate_sphere.run("s", ico, cfg)
    assert not res.ok
    assert {"nonfinite_P", "off_unit_sphere", "coverage_not_single", "not_centered"} <= _codes(res)
    assert res.meta["n_nonfinite_P"] == 1


def test_rejects_one_flipped_triangle(cfg, ico):
    P, F = ico["P"], ico["F"]
    v, a, b = F[0]
    m = (P[a] + P[b]) / 2
    q = m + 0.1 * (m - P[v])  # push vertex v just past its link edge (a, b)
    P[v] = q / np.linalg.norm(q)
    res = validate_sphere.run("s", ico, cfg)
    assert not res.ok
    assert "flipped_triangles" in _codes(res)
    assert res.meta["n_flipped"] == 1
    assert abs(res.meta["coverage_err"]) < 1e-12  # degree is preserved; only the det check sees it


def test_rejects_degenerate_triangle(cfg, ico):
    P, F = ico["P"], ico["F"]
    v, a, _ = F[0]
    P[v] = P[a]  # collapse an edge on the sphere
    res = validate_sphere.run("s", ico, cfg)
    assert not res.ok
    assert "degenerate_sphere_triangles" in _codes(res)
    assert res.meta["n_degenerate"] == 2
    assert res.meta["min_sphere_edge"] == 0.0


def test_rejects_permuted_rows(cfg, ico):
    perm = np.arange(len(ico["P"]))
    perm[[3, 40]] = perm[[40, 3]]
    ico["P"], ico["X"] = ico["P"][perm], ico["X"][perm]  # output read back in the wrong order
    res = validate_sphere.run("s", ico, cfg)
    assert not res.ok
    assert "correspondence_mismatch" in _codes(res)
    assert res.meta["correspondence_ok"] is False


def test_rejects_float64_x_that_differs_from_float32_submission(cfg, ico):
    ico["X"] = ico["X"].astype(np.float64) + 1e-7
    res = validate_sphere.run("s", ico, cfg)
    assert _codes(res) == {"correspondence_mismatch"}


def test_rejects_mismatched_faces(cfg, ico):
    F_sphere = ico["F"].copy()
    F_sphere[0] = F_sphere[0, ::-1]
    ico["F_sphere"] = F_sphere
    res = validate_sphere.run("s", ico, cfg)
    assert _codes(res) == {"correspondence_mismatch"}


def test_rejects_off_sphere(cfg, ico):
    ico["P"] = ico["P"] * (1 + 1e-6)
    res = validate_sphere.run("s", ico, cfg)
    assert not res.ok
    assert "off_unit_sphere" in _codes(res)
    np.testing.assert_allclose(res.meta["max_norm_err"], 1e-6, rtol=1e-6)


def test_rejects_uncentred_map(cfg, ico):
    P = ico["P"] + [0, 0, 0.3]
    ico["P"] = P / np.linalg.norm(P, axis=1, keepdims=True)
    res = validate_sphere.run("s", ico, cfg)
    assert _codes(res) == {"not_centered"}
    assert res.meta["centering_residual"] > 0.05


def test_missing_input(cfg, ico):
    del ico["X"]
    res = validate_sphere.run("s", ico, cfg)
    assert not res.ok and res.reason.startswith("missing_input")



def test_robust_orientation_on_real_false_flips():
    """Regression: 134 tip triangles (edges ~4e-11) from a valid L/r = 20 SphereMap output.
    det[p0,p1,p2] calls every one of them flipped; the robust orientation calls none flipped."""
    from moe_data import geometry as geo
    T = np.load(Path(__file__).parent / "data" / "tip_false_flip_triangles.npz")["T"]
    P = T.reshape(-1, 3)
    F = np.arange(len(P)).reshape(-1, 3)
    assert np.all(geo.spherical_triangle_orientation(P, F) <= 0)   # the failure being guarded against
    assert np.all(geo.robust_spherical_orientation(P, F) > 0)
