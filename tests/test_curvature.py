import copy
from pathlib import Path

import numpy as np
import pytest
import trimesh
from numpy.testing import assert_allclose

from moe_data.config import load_config
from moe_data.stages import curvature

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return copy.deepcopy(load_config(CONFIG))


@pytest.fixture
def ellipsoid():
    m = trimesh.creation.icosphere(subdivisions=3)
    V = np.asarray(m.vertices, dtype=np.float64) * [1.0, 0.7, 0.5]
    return V, np.asarray(m.faces, dtype=np.int64)


def _torus(nu=30, nv=15, R=1.0, r=0.3):
    u, v = np.meshgrid(np.arange(nu) * 2 * np.pi / nu, np.arange(nv) * 2 * np.pi / nv, indexing="ij")
    V = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], -1)
    idx = lambda i, j: (i % nu) * nv + (j % nv)  # noqa: E731
    F = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)] for i in range(nu) for j in range(nv)]
    F += [[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(nu) for j in range(nv)]
    return V.reshape(-1, 3), np.array(F, dtype=np.int64)


def test_accepts_ellipsoid_and_outputs_finite(cfg, ellipsoid):
    V, F = ellipsoid
    res = curvature.run("s", {"V": V, "F": F}, cfg)
    assert res.ok, res.reason
    for k in ("area", "K_int", "K_pt", "H"):
        assert res.data[k].shape == (len(V),)
        assert np.all(np.isfinite(res.data[k]))
    assert_allclose(res.data["K_pt"], res.data["K_int"] / res.data["area"], rtol=1e-15)
    assert res.meta["gauss_bonnet_err"] < 1e-10
    assert res.meta["H_q00"] > 0 and res.meta["K_pt_q00"] > 0
    assert res.meta["K_pt_q00"] <= res.meta["K_pt_q50"] <= res.meta["K_pt_q100"]


def test_prefers_submitted_vertices(cfg, ellipsoid):
    V, F = ellipsoid
    res = curvature.run("s", {"V": 2 * V, "V_sub": V, "F": F}, cfg)
    ref = curvature.run("s", {"V": V, "F": F}, cfg)
    assert_allclose(res.data["H"], ref.data["H"], rtol=1e-15)


def test_rejects_torus_gauss_bonnet(cfg):
    V, F = _torus()
    res = curvature.run("s", {"V": V, "F": F}, cfg)
    assert not res.ok
    assert res.reason.startswith("gauss_bonnet")
    assert_allclose(res.meta["gauss_bonnet_err"], 4 * np.pi, rtol=1e-10)


def test_rejects_nan_vertex(cfg, ellipsoid):
    V, F = ellipsoid
    V = V.copy()
    V[7] = np.nan
    res = curvature.run("s", {"V": V, "F": F}, cfg)
    assert not res.ok
    assert "nonfinite_K_int" in res.reason and "gauss_bonnet" in res.reason
    assert np.isnan(res.meta["gauss_bonnet_err"])


def test_rejects_zero_area_vertex(cfg, ellipsoid):
    V, F = ellipsoid
    V = np.vstack([V, [[0.0, 0, 0]]])  # unreferenced vertex: zero area -> K_pt not finite
    res = curvature.run("s", {"V": V, "F": F}, cfg)
    assert not res.ok
    assert "nonfinite_K_pt" in res.reason


def test_missing_input(cfg):
    res = curvature.run("s", {"F": np.zeros((1, 3), dtype=np.int64)}, cfg)
    assert not res.ok and res.reason.startswith("missing_input")
