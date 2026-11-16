from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from moe_data.config import load_config
from moe_data.dedup import (aligned_shape, canonical_faces, chamfer, dedup_records, descriptor, exact_hash,
                            group_duplicates)

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


# ---- face-order independent descriptor, Chamfer confirmation, two-stage grouping

def blob(sub, scale=(1.0, 0.7, 0.45), bump=(0.6, 0.0, 0.0)):
    """Closed asymmetric test shape: anisotropic sphere with a bump towards +x, triangulated at
    icosphere level ``sub`` (a different level is a re-triangulated copy of the same surface)."""
    m = trimesh.creation.icosphere(sub)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    r = 1.0 + bump[0] * np.exp(-((u[:, 0] - 1.0) ** 2 + (u[:, 1] - bump[1]) ** 2 + (u[:, 2] - bump[2]) ** 2) / 0.1)
    return u * r[:, None] * np.asarray(scale), m.faces.astype(np.int64)


def chamfer_threshold():
    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")["dedup"]["chamfer_threshold"]


def test_descriptor_invariant_to_face_and_vertex_order():
    V, F = blob(3)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(V))
    V2, F2 = V[perm], np.argsort(perm)[F]                       # vertices renumbered
    F2 = np.roll(F2, 1, axis=1)[rng.permutation(len(F2))]       # faces rotated and shuffled
    np.testing.assert_allclose(desc(V2, F2), desc(V, F), rtol=0, atol=1e-12)


def test_descriptor_seed_noise_below_candidate_scale():
    V, F = blob(3)
    d = [descriptor(V, F, np.random.default_rng(s)) for s in (1, 2, 3)]
    assert max(np.linalg.norm(d[0] - d[k]) for k in (1, 2)) < 0.003


def test_canonical_faces_same_triangles():
    V, F = blob(2)
    G = canonical_faces(V, F)
    assert exact_hash(V, G, 9) == exact_hash(V, F, 9)
    np.testing.assert_array_equal(G, canonical_faces(V, F[::-1]))


def test_chamfer_confirms_rotated_retriangulated_copy_and_rejects_other_shape():
    thr = chamfer_threshold()
    V, F = blob(4)
    Vr, Fr = blob(3)                                            # same surface, coarser triangulation
    R = Rotation.from_euler("xyz", [40, -70, 115], degrees=True).as_matrix()
    Vr = 2.5 * Vr @ R.T + [3.0, -1.0, 0.5]                      # rotated, scaled, moved
    Vo, Fo = blob(4, scale=(1.0, 0.8, 0.35), bump=(0.6, 0.7, 0.0))   # different shape
    a = aligned_shape(V, F, np.random.default_rng(0))
    b = aligned_shape(Vr, Fr, np.random.default_rng(1))
    o = aligned_shape(Vo, Fo, np.random.default_rng(2))
    same = chamfer(a, b)
    assert same < 0.3 * thr
    assert chamfer(b, a) == pytest.approx(same, abs=0.2 * thr)
    assert chamfer(a, o) > 2 * thr
    # identical surface: only ICP convergence error remains, far below the remesh self-noise (~1e-3)
    assert chamfer(a, aligned_shape(V, F[::-1], np.random.default_rng(5))) < 1e-4


def test_chamfer_symmetric_axis_rotated_in_plane():
    """Two equal principal extents: the in-plane orientation is not fixed by PCA and is searched."""
    V, F = blob(4, scale=(1.0, 1.0, 0.5), bump=(0.5, 0.0, 0.0))
    R = Rotation.from_euler("z", 37, degrees=True).as_matrix()
    a = aligned_shape(V, F, np.random.default_rng(0))
    b = aligned_shape(V @ R.T, F, np.random.default_rng(1))
    assert chamfer(a, b) < 1e-4


def test_confirm_merges_pair_the_descriptor_puts_at_0_008():
    ids, hashes, things, nf = ["a", "b"], ["h1", "h2"], [1, 2], [10, 20]
    descs = [np.zeros(4), np.array([0.008, 0.0, 0.0, 0.0])]
    yes = dedup_records(ids, descs, hashes, things, THRESH, nf, confirm=lambda c: [True] * len(c))
    assert yes["a"]["group_id"] == yes["b"]["group_id"] and yes["a"]["dup_group_reason"] == "near"
    assert yes["b"]["keep"] and yes["a"]["reject_reason"] == "duplicate_of: b"
    no = dedup_records(ids, descs, hashes, things, THRESH, nf, confirm=lambda c: [False] * len(c))
    assert no["a"]["group_id"] != no["b"]["group_id"] and no["a"]["keep"] and no["b"]["keep"]


def test_confirm_sees_only_candidates_below_threshold():
    seen = []

    def confirm(cand):
        seen.extend(map(tuple, cand))
        return [True] * len(cand)
    descs = [np.zeros(2), np.array([0.01, 0.0]), np.array([1.0, 0.0])]
    recs = dedup_records(["a", "b", "c"], descs, ["x", "y", "z"], [1, 2, 3], THRESH, [1, 1, 1], confirm)
    assert seen == [(0, 1)]
    assert recs["c"]["dup_group_reason"] == ""
