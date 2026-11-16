import numpy as np
import trimesh

from moe_data.dedup import dedup_records, descriptor, exact_hash, group_duplicates

THRESH = 0.02


def sphere(sub=3):
    m = trimesh.creation.icosphere(sub)
    return m.vertices.copy(), m.faces.astype(np.int64)


def desc(V, F):
    return descriptor(V, F, np.random.default_rng(1234))


def test_exact_hash_invariant_to_vertex_permutation_and_face_rotation():
    V, F = sphere(2)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(V))
    inv = np.argsort(perm)
    V2, F2 = V[perm], inv[F]                       # same mesh, vertices renumbered
    F2 = np.roll(F2, 1, axis=1)                    # rotate every face (orientation kept)
    F2 = F2[rng.permutation(len(F2))]              # shuffle faces
    assert exact_hash(V, F, 5) == exact_hash(V2, F2, 5)


def test_exact_hash_sensitive_to_geometry_and_orientation():
    V, F = sphere(2)
    h = exact_hash(V, F, 5)
    assert h != exact_hash(V + [1e-3, 0, 0], F, 5)
    assert h != exact_hash(V, F[:, ::-1], 5)
    assert h == exact_hash(V + 1e-8, F, 5)


def test_descriptor_deterministic_and_finite():
    V, F = sphere()
    d = desc(V, F)
    assert np.isfinite(d).all()
    np.testing.assert_array_equal(d, desc(V, F))
    # scale and translation invariant (normalised internally)
    np.testing.assert_allclose(desc(3 * V + 5, F), d, atol=1e-2)


def test_near_duplicate_grouped_but_sphere_and_cube_not():
    V, F = sphere()
    Vn = V + 1e-4 * np.random.default_rng(7).standard_normal(V.shape)
    b = trimesh.creation.box()
    ids = ["a", "b", "c"]
    descs = [desc(V, F), desc(Vn, F), desc(b.vertices, b.faces)]
    hashes = [exact_hash(V, F, 5), exact_hash(Vn, F, 5), exact_hash(b.vertices, b.faces, 5)]
    assert np.linalg.norm(descs[0] - descs[1]) < THRESH
    assert np.linalg.norm(descs[0] - descs[2]) > THRESH
    groups = group_duplicates(ids, descs, hashes, [1, 2, 3], THRESH, n_faces=[10, 10, 12])
    assert groups["a"] == groups["b"] != groups["c"]
    recs = dedup_records(ids, descs, hashes, [1, 2, 3], THRESH, [10, 10, 12])
    assert recs["a"]["keep"] and recs["a"]["dup_group_reason"] == "near"
    assert not recs["b"]["keep"] and recs["b"]["reject_reason"] == "duplicate_of: a"
    assert recs["b"]["reject_stage"] == "dedup"
    assert recs["c"]["keep"] and recs["c"]["dup_group_reason"] == ""


def test_representative_largest_faces_and_same_thing_not_rejected():
    ids = ["p", "q", "r", "s"]
    descs = [np.zeros(4), np.zeros(4) + 1.0, np.zeros(4) + 2.0, np.zeros(4) + 3.0]
    hashes = ["h1", "h1", "h2", "h3"]      # p,q exact duplicates
    things = [10, 11, 12, 12]              # r,s same thing, different shapes
    recs = dedup_records(ids, descs, hashes, things, THRESH, [100, 200, 5, 5])
    assert recs["q"]["keep"] and recs["p"]["reject_reason"] == "duplicate_of: q"
    assert recs["p"]["dup_group_reason"] == "exact"
    assert recs["r"]["keep"] and recs["s"]["keep"]
    assert recs["r"]["group_id"] == recs["s"]["group_id"] == "r"
    assert recs["r"]["dup_group_reason"] == "thing"
    assert recs["p"]["group_id"] == recs["q"]["group_id"] == "p"
    # tie in face count -> smallest id kept
    recs = dedup_records(["b", "a"], [np.zeros(2), np.zeros(2)], ["x", "y"], [1, 2], THRESH, [5, 5])
    assert recs["a"]["keep"] and recs["b"]["reject_reason"] == "duplicate_of: a"
    assert recs["b"]["dup_group_reason"] == "near"


def test_empty_and_single():
    assert group_duplicates([], np.zeros((0, 4)), [], [], THRESH) == {}
    assert group_duplicates(["a"], [np.zeros(4)], ["h"], [1], THRESH) == {"a": "a"}
