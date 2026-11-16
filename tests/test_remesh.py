import copy
from pathlib import Path

import numpy as np
import numpy.testing as npt
import trimesh

from moe_data import procedural
from moe_data.config import load_config
from moe_data.stages import normalize, remesh
from moe_data.stages.suitability import surface_stats
from moe_data.topology import check_topology

CFG = load_config(Path(__file__).resolve().parents[1] / "config.yaml")


def code(reason):
    return reason.split(":")[0]


def normalised(V, F):
    r = normalize.run("t", {"V": V, "F": F}, CFG)
    assert r.ok
    return r.data


def revolve(rho, z, n_phi):
    """Closed surface of revolution; profile starts and ends on the axis (rho = 0)."""
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    rings = [np.c_[r * np.cos(phi), r * np.sin(phi), np.full(n_phi, zz)]
             for r, zz in zip(rho[1:-1], z[1:-1])]
    V = np.vstack([[0, 0, z[0]], *rings, [0, 0, z[-1]]])
    nr = len(rings)
    j = np.arange(n_phi)
    jn = (j + 1) % n_phi
    F = [np.c_[np.zeros(n_phi, int), 1 + j, 1 + jn]]
    for k in range(nr - 1):
        a = 1 + k * n_phi
        b = a + n_phi
        F += [np.c_[a + j, b + j, b + jn], np.c_[a + j, b + jn, a + jn]]
    s, a = len(V) - 1, 1 + (nr - 1) * n_phi
    F.append(np.c_[np.full(n_phi, s), a + jn, a + j])
    return V.astype(np.float64), np.vstack(F).astype(np.int64)


def sphere_with_spike(w=0.05, length=2.0, n_phi=32):
    """Unit sphere with a thin cone of base radius w reaching z = length."""
    zb = np.sqrt(1 - w * w)
    zt = np.linspace(length, zb, 80)[1:]
    th = np.linspace(np.arcsin(w), np.pi, 120)
    rho = np.r_[0, w * (length - zt) / (length - zb), np.sin(th)[1:]]
    z = np.r_[length, zt, np.cos(th)[1:]]
    return revolve(rho, z, n_phi)


def assert_good_remesh(res):
    assert res.ok, res.reason
    V, F = res.data["V"], res.data["F"]
    assert V.dtype == np.float64 and F.dtype == np.int64
    assert CFG["remesh"]["min_vertices"] <= len(V) <= CFG["remesh"]["max_vertices"]
    st = surface_stats(V, F)
    npt.assert_allclose(st["area"], 1.0, rtol=1e-9)
    npt.assert_allclose(st["centroid"], 0.0, atol=1e-12)
    ok, reason, _ = check_topology(V, F)
    assert ok, reason
    m = res.meta
    assert m["hausdorff_sym"] == max(m["hausdorff_remesh_to_orig"], m["hausdorff_orig_to_remesh"])
    assert m["hausdorff_sym"] < 0.1 * CFG["remesh"]["hausdorff_tol"]
    npt.assert_allclose(m["remesh_area_ratio"], 1.0, atol=5e-3)
    assert m["self_intersecting_remesh"] is False
    assert m["remesh_min_tri_area"] > 0 and m["remesh_min_angle_deg"] > 0
    assert m["remesh_n_unreferenced"] == 0


def test_target_edge_length_equilateral_count():
    # equilateral triangulation, F = 2V: V * 2 * sqrt(3)/4 h^2 = A
    h = remesh.target_edge_length(1.0, 15000)
    npt.assert_allclose(15000 * 2 * np.sqrt(3) / 4 * h**2, 1.0, rtol=1e-12)
    npt.assert_allclose(remesh.target_edge_length(4.0, 15000), 2 * h, rtol=1e-12)


def test_icosphere_remesh_lands_in_range():
    m = trimesh.creation.icosphere(6)
    res = remesh.run("ico6", normalised(m.vertices, m.faces), CFG)
    assert_good_remesh(res)
    assert res.meta["remesh_attempts"] == 1   # the edge-length formula is right without iteration


def test_protrusion_remesh_lands_in_range():
    V, F, _ = procedural.generate("protrusion", 0, CFG)
    assert_good_remesh(remesh.run("proc_prot_0", normalised(V, F), CFG))


def test_spike_lost_only_in_orig_to_remesh_direction():
    V, F = sphere_with_spike()
    assert check_topology(V, F)[0]
    res = remesh.run("spike", normalised(V, F), CFG)
    tol = CFG["remesh"]["hausdorff_tol"]
    assert not res.ok and code(res.reason) == "remesh_hausdorff"
    # a one-sided remesh->orig check would have accepted this mesh
    assert res.meta["hausdorff_remesh_to_orig"] < tol
    assert res.meta["hausdorff_orig_to_remesh"] > 5 * tol
    assert "V" not in res.data


def test_vertex_count_out_of_range_rejects():
    cfg = copy.deepcopy(CFG)
    cfg["remesh"].update(target_vertices=500, min_vertices=10, max_vertices=20, max_attempts=2)
    m = trimesh.creation.icosphere(3)
    res = remesh.run("ico3", normalised(m.vertices, m.faces), cfg)
    assert not res.ok and code(res.reason) == "remesh_vertex_count"
    assert res.meta["remesh_attempts"] == 2


def test_vertex_count_converges_from_bad_initial_count():
    # a target far off the achievable count must be corrected by the h update
    cfg = copy.deepcopy(CFG)
    cfg["remesh"].update(target_vertices=2000, min_vertices=1900, max_vertices=2100, max_attempts=4)
    m = trimesh.creation.icosphere(4)
    res = remesh.run("ico4", normalised(m.vertices, m.faces), cfg)
    assert res.ok, res.reason
    assert 1900 <= len(res.data["V"]) <= 2100


def test_nonfinite_input_rejects():
    m = trimesh.creation.icosphere(2)
    V = m.vertices.copy()
    V[0] = np.nan
    res = remesh.run("nan", {"V": V, "F": m.faces}, CFG)
    assert not res.ok and code(res.reason) == "remesh_bad_input"


def test_missing_input_rejects():
    res = remesh.run("x", {}, CFG)
    assert not res.ok and code(res.reason) == "missing_input"
