"""visibility stage: view fields against the analytic sphere cap, occlusion, paint kernels, determinism."""
import copy
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.stats import spearmanr

from moe_data import geometry as geo
from moe_data.config import load_config
from moe_data.stages import visibility as vs

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return copy.deepcopy(load_config(CONFIG))


def _sphere(subdiv=5, radius=1.0, centre=(0.0, 0.0, 0.0)):
    m = trimesh.creation.icosphere(subdivisions=subdiv)
    V = np.asarray(m.vertices, np.float64)
    V = radius * V / np.linalg.norm(V, axis=1, keepdims=True) + np.asarray(centre)
    return V, np.asarray(m.faces, np.int64)


def _unit_area(V, F):
    return V / np.sqrt(geo.face_areas(V, F).sum()), F


def test_sphere_view_matches_perspective_cap(cfg):
    V, F = _sphere()
    d = np.array([0.3, -0.5, 0.8])
    d /= np.linalg.norm(d)
    cv = cfg["visibility"]
    mask = vs.visible_mask(V, F, d, cv)
    area = geo.vertex_areas_barycentric(V, F)
    # camera distance D = camera_distance * bbox half-diagonal = camera_distance * sqrt(3) R;
    # a pinhole at distance D sees the cap with cos(theta) = R/D, area fraction (1 - R/D)/2.
    D = cv["camera_distance"] * np.sqrt(3.0)
    np.testing.assert_allclose(area[mask].sum() / area.sum(), 0.5 * (1 - 1 / D), atol=0.01)
    ndot = geo.vertex_normals(V, F) @ d
    assert np.all(mask[ndot > 1 / D + 0.05]) and not np.any(mask[ndot < 1 / D - 0.05])


def test_sphere_view_field_correlates_with_normal(cfg):
    V, F = _unit_area(*_sphere())
    r = vs.run("sphere", {"V_sub": V, "F": F}, cfg)
    assert r.ok, r.reason
    N = geo.vertex_normals(V, F)
    for j in np.flatnonzero(r.data["vis_kind"] == "view"):
        d = r.data["vis_dirs"][j]
        assert np.corrcoef(r.data["vis"][:, j], N @ d)[0, 1] > 0.8
        np.testing.assert_allclose(r.meta["vis_frac_visible"][j], 0.5, atol=0.07)


def test_occlusion_hides_front_facing_vertices(cfg):
    """Two spheres on the x axis; the camera on +x sees the near one, which hides the middle of the far one."""
    Va, Fa = _sphere(4, 1.0, (0.0, 0.0, 0.0))
    Vb, Fb = _sphere(4, 1.0, (3.0, 0.0, 0.0))
    V, F = np.vstack([Va, Vb]), np.vstack([Fa, Fb + len(Va)])
    N = geo.vertex_normals(V, F)
    mask = vs.visible_mask(V, F, np.array([1.0, 0.0, 0.0]), cfg["visibility"], N)
    far = np.arange(len(V)) < len(Va)
    front_far = far & (N[:, 0] > 0.5)
    hidden = front_far & ~mask
    assert hidden.sum() > 0
    # the far sphere's front pole (1, 0, 0) is directly behind the near sphere
    assert not mask[np.argmax(np.where(far, V[:, 0], -np.inf))]
    # nothing occludes the near sphere's front pole
    assert mask[np.argmax(V[:, 0])]


def _trimesh_visible(V, F, d, cv):
    """Oracle: trimesh's ray engine on the same rays (origin offset along n, towards the camera)."""
    lo, hi = V.min(axis=0), V.max(axis=0)
    radius = 0.5 * np.linalg.norm(hi - lo)
    cam = 0.5 * (lo + hi) + cv["camera_distance"] * radius * d
    N = geo.vertex_normals(V, F)
    front = np.einsum("ij,ij->i", N, cam - V) > 0
    o = V + cv["ray_offset_rel"] * radius * N
    dirs = (cam - o) / np.linalg.norm(cam - o, axis=1, keepdims=True)
    hit = trimesh.ray.ray_triangle.RayMeshIntersector(trimesh.Trimesh(V, F, process=False)).intersects_any(o, dirs)
    return front & ~hit


def test_occlusion_matches_trimesh_rays(cfg):
    Va, Fa = _sphere(3, 1.0, (0.0, 0.0, 0.0))
    Vb, Fb = _sphere(3, 0.6, (1.8, 0.4, 0.0))
    V, F = np.vstack([Va, Vb]), np.vstack([Fa, Fb + len(Va)])
    for d in (np.array([1.0, 0.0, 0.0]), np.array([0.6, 0.8, 0.0]), np.array([0.0, 0.0, -1.0])):
        mine = vs.visible_mask(V, F, d, cfg["visibility"])
        ref = _trimesh_visible(V, F, d, cfg["visibility"])
        np.testing.assert_array_equal(mine, ref)
        assert (~mine).sum() > len(V) // 2 - 0.1 * len(V)      # a non-trivial mask


def test_paint_peaks_at_seed_and_decays(cfg):
    V, F = _unit_area(*_sphere())
    solve, M = vs._heat_solver(V, F, cfg["visibility"]["paint_diffusion_t"])
    seed = 123
    w = vs.paint_field(solve, M, np.array([seed]))
    np.testing.assert_allclose(w[seed], 1.0, atol=1e-12)
    assert w.min() >= 0.0 and w.max() <= 1.0
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    angle = np.arccos(np.clip(U @ U[seed], -1, 1))
    assert spearmanr(angle, w).statistic < -0.99


def test_paint_two_seeds_both_one(cfg):
    V, F = _unit_area(*_sphere(4))
    solve, M = vs._heat_solver(V, F, cfg["visibility"]["paint_diffusion_t"])
    seeds = np.array([0, 7])
    w = vs.paint_field(solve, M, seeds)
    np.testing.assert_allclose(w[seeds], 1.0, atol=1e-12)


def test_run_outputs_shapes_range_and_determinism(cfg):
    V, F = _unit_area(*_sphere(4))
    a = vs.run("proc_blob_1", {"V_sub": V, "F": F}, cfg)
    b = vs.run("proc_blob_1", {"V_sub": V, "F": F}, cfg)
    c = vs.run("proc_blob_2", {"V_sub": V, "F": F}, cfg)
    cv = cfg["visibility"]
    n_view, n_paint = cv["n_view"], cv["n_paint"]
    assert a.ok and a.data["vis"].dtype == np.float32
    assert a.data["vis"].shape == (len(V), n_view + n_paint)
    assert list(a.data["vis_kind"]) == ["view"] * n_view + ["paint"] * n_paint
    assert a.data["vis_dirs"].shape == (n_view, 3)
    np.testing.assert_allclose(np.linalg.norm(a.data["vis_dirs"], axis=1), 1.0, atol=1e-12)
    assert np.all(np.isfinite(a.data["vis"])) and a.data["vis"].min() >= 0 and a.data["vis"].max() <= 1
    np.testing.assert_array_equal(a.data["vis"], b.data["vis"])
    assert a.meta == {**b.meta, "vis_raycast_s": a.meta["vis_raycast_s"]}
    assert not np.array_equal(a.data["vis"], c.data["vis"])
    for seeds in a.meta["vis_paint_seeds"]:
        assert cv["paint_blobs"][0] <= len(seeds) <= cv["paint_blobs"][1]
    import json
    json.dumps(a.meta)                                           # meta is JSON-able


def test_missing_input(cfg):
    r = vs.run("x", {"F": np.zeros((0, 3), np.int64)}, cfg)
    assert not r.ok and r.reason.startswith("missing_input")


def test_timing_15k(cfg):
    m = trimesh.creation.uv_sphere(count=[87, 87])        # 15138 vertices
    V, F = _unit_area(np.asarray(m.vertices, np.float64) * [1.0, 0.7, 0.5], np.asarray(m.faces, np.int64))
    t0 = time.perf_counter()
    r = vs.run("timing", {"V_sub": V, "F": F}, cfg)
    dt = time.perf_counter() - t0
    print(f"\nvisibility on {len(V)} vertices: {dt:.2f} s total, ray cast {r.meta['vis_raycast_s']:.2f} s")
    assert r.ok, r.reason
    assert dt < 2.0
