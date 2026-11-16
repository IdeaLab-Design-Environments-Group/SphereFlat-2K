import numpy as np
import numpy.testing as npt
import trimesh

from moe_data.stages import normalize, suitability
from moe_data.stages import topology as topology_stage
from moe_data.topology import check_and_orient, check_topology


def sphere(sub=2, center=(0.0, 0.0, 0.0)):
    m = trimesh.creation.icosphere(sub)
    return m.vertices + np.asarray(center), m.faces.astype(np.int64)


def code(reason):
    return reason.split(":")[0]


def test_icosphere_passes():
    V, F = sphere()
    ok, reason, m = check_topology(V, F)
    assert ok and reason == ""
    assert m["chi"] == 2 and m["genus"] == 0
    assert m["n_vertices"] == len(V) and m["n_faces"] == len(F)
    assert m["flipped_orientation"] is False


def test_torus_rejected_not_genus0():
    t = trimesh.creation.torus(1.0, 0.3)
    ok, reason, m = check_topology(t.vertices, t.faces)
    assert not ok and code(reason) == "not_genus0"
    assert m["chi"] == 0 and m["genus"] == 1


def test_two_spheres_rejected_multiple_components():
    V1, F1 = sphere()
    V2, F2 = sphere(center=(5, 0, 0))
    ok, reason, m = check_topology(np.vstack([V1, V2]), np.vstack([F1, F2 + len(V1)]))
    assert not ok and code(reason) == "multiple_components"
    assert m["n_components"] == 2 and m["chi"] == 4


def test_open_sphere_rejected_not_closed():
    V, F = sphere()
    ok, reason, m = check_topology(V, F[1:])
    assert not ok and code(reason) == "not_closed"
    assert m["n_boundary_edges"] == 3


def test_bowtie_rejected_vertex_nonmanifold():
    c = trimesh.creation.cone(1.0, 1.0, sections=12)
    V1, F1 = c.vertices, c.faces.astype(np.int64)
    apex = int(np.argmax(V1[:, 2]))
    assert np.allclose(V1[apex], [0, 0, 1])
    # rotate 180 deg about x (proper rotation keeps orientation), apex lands on the same point
    V2 = V1 * [1, -1, -1] + [0, 0, 2]
    F2 = F1 + len(V1)
    F2[F2 == apex + len(V1)] = apex
    V = np.vstack([V1, V2])
    ok, reason, m = check_topology(V, np.vstack([F1, F2]))
    assert not ok and code(reason) == "vertex_nonmanifold"
    assert m["n_nonmanifold_vertices"] == 1


def test_three_faces_on_edge_rejected_edge_nonmanifold():
    # closed octahedron plus an internal membrane spanning its equator square:
    # every equator edge then has 3 faces, and there are no boundary edges
    V = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    F = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [1, 0, 5], [2, 1, 5], [3, 2, 5], [0, 3, 5],
         [0, 1, 2], [0, 2, 3]]
    ok, reason, m = check_topology(V, np.array(F))
    assert not ok and code(reason) == "edge_nonmanifold"
    assert m["n_boundary_edges"] == 0 and m["n_nonmanifold_edges"] == 4


def test_one_flipped_face_rejected_inconsistent_orientation():
    V, F = sphere()
    F = F.copy()
    F[0] = F[0, ::-1]
    ok, reason, m = check_topology(V, F)
    assert not ok and code(reason) == "inconsistent_orientation"
    assert m["n_inconsistent_edges"] == 3
    assert m["n_nonmanifold_vertices"] == 0


def test_inverted_mesh_passes_and_is_flipped_outward():
    V, F = sphere()
    ok, reason, m, V2, F2 = check_and_orient(V, F[:, ::-1])
    assert ok and m["flipped_orientation"] is True
    a, b, c = V2[F2[:, 0]], V2[F2[:, 1]], V2[F2[:, 2]]
    normals = np.cross(b - a, c - a)
    assert (np.einsum("ij,ij->i", normals, (a + b + c) / 3) > 0).all()
    assert F2.dtype == np.int64


def test_degenerate_inputs():
    V, F = sphere(0)
    cases = [
        (V, np.zeros((0, 3), np.int64)),
        (np.where(np.arange(len(V))[:, None] == 0, np.nan, V), F),
        (V, np.where(F == F.max(), len(V), F)),
        (V, np.vstack([F, [[0, 0, 1]]])),
        (V[:, :2], F),
    ]
    for Vc, Fc in cases:
        ok, reason, m = check_topology(Vc, Fc)
        assert not ok and code(reason) == "degenerate_input", reason


def test_unreferenced_vertex_removed_and_duplicates_rejected():
    V, F = sphere(1)
    ok, _, m, V2, _ = check_and_orient(np.vstack([V, [[9, 9, 9]]]), F)
    assert ok and m["n_unreferenced_removed"] == 1 and len(V2) == len(V)
    # split one vertex into a coincident copy used by one face
    Vd = np.vstack([V, V[F[0, 0]]])
    Fd = F.copy()
    Fd[0, 0] = len(V)
    ok, reason, m = check_topology(Vd, Fd)
    assert not ok and code(reason) == "duplicate_vertices"
    assert m["n_duplicate_positions"] == 1


def test_topology_stage_contract():
    V, F = sphere()
    r = topology_stage.run("x", {"V": V, "F": F[:, ::-1]}, {})
    assert r.ok and r.meta["flipped_orientation"] and r.data["F"].shape == F.shape
    r = topology_stage.run("x", {"V": V, "F": F[1:]}, {})
    assert not r.ok and r.reason.startswith("not_closed: ")
    r = topology_stage.run("x", {}, {})
    assert not r.ok and r.reason.startswith("missing_input: ")


def test_normalize_unit_area_zero_centroid():
    V, F = sphere(3)
    V = V * [2.0, 1.0, 0.5] + [3.0, -1.0, 7.0]
    r = normalize.run("x", {"V": V, "F": F}, {})
    assert r.ok
    st = suitability.surface_stats(r.data["V"], r.data["F"])
    npt.assert_allclose(st["area"], 1.0, rtol=1e-12)
    npt.assert_allclose(st["centroid"], 0.0, atol=1e-14)
    npt.assert_allclose(r.meta["scale"], 1 / np.sqrt(r.meta["orig_area"]), rtol=1e-15)


def test_surface_stats_sphere_and_box():
    V, F = sphere(5)
    m = suitability.measures(V, F)
    npt.assert_allclose(m["compactness"], 1 / (6 * np.sqrt(np.pi)), rtol=1e-3)  # 0.0940
    npt.assert_allclose(m["pca_ratio_min"], 1.0, rtol=1e-6)
    b = trimesh.creation.box(extents=[1, 2, 4])
    st = suitability.surface_stats(b.vertices, b.faces)
    # uniform-on-surface second moment of a box, evaluated per face analytically: ext ratios
    assert st["extents"][0] < st["extents"][1] < st["extents"][2]
    npt.assert_allclose(st["volume"], 8.0, rtol=1e-12)


def test_suitability_rejects_and_records_measures():
    cfg = {"suitability": {"min_faces": 500, "min_compactness": 0.005, "min_pca_ratio": 0.02}}
    V, F = sphere(3)
    assert suitability.run("x", {"V": V, "F": F}, cfg).ok
    r = suitability.run("x", {"V": V, "F": F}, {"suitability": {**cfg["suitability"], "min_faces": 10**6}})
    assert not r.ok and r.reason.startswith("too_few_faces: ")
    assert set(r.meta) >= {"n_faces", "compactness", "pca_ratio_min", "pca_ratio_mid"}
    flat = V * [1, 1, 0.005]
    r = suitability.run("x", {"V": flat, "F": F}, cfg)
    assert not r.ok and r.reason.split(":")[0] in {"not_compact", "too_thin"}
    # flattened sphere = two-sided disc: z-rms/x-rms = 0.005 * sqrt(0.5 / 0.25)
    npt.assert_allclose(r.meta["pca_ratio_min"], 0.005 * np.sqrt(2), rtol=0.02)
