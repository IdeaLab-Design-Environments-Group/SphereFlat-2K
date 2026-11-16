import os
import subprocess
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import trimesh

from moe_data.io_ply import read_spheremap_ply, write_ply

ROOT = Path(__file__).resolve().parents[1]
SPHEREMAP = ROOT / "third_party/MoebiusRegistration/Bin/Linux/SphereMap"


def _write_spheremap_like(path, X, P, F, fmt="binary_little_endian"):
    """Write a PLY with SphereMap's output layout (float xyz, double pxyz, uchar rgb)."""
    n = len(X)
    header = (
        f"ply\nformat {fmt} 1.0\ncomment made by test\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property double px\nproperty double py\nproperty double pz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        if fmt == "ascii":
            for x, p in zip(X, P):
                f.write((" ".join(map(repr, [*x.astype(np.float32).tolist(), *p.tolist()])) + " 1 2 3\n").encode())
            for t in F:
                f.write(f"3 {t[0]} {t[1]} {t[2]}\n".encode())
            return
        e = "<" if fmt == "binary_little_endian" else ">"
        v = np.empty(n, dtype=[("x", e + "f4", (3,)), ("p", e + "f8", (3,)), ("c", "u1", (3,))])
        v["x"], v["p"], v["c"] = X, P, 7
        fc = np.empty(len(F), dtype=[("n", "u1"), ("i", e + "i4", (3,))])
        fc["n"], fc["i"] = 3, F
        f.write(v.tobytes())
        f.write(fc.tobytes())


def _random_mesh(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(7, 3))
    P = rng.normal(size=(7, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    F = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 0, 3]])
    return X, P, F


def test_write_ply_header_and_payload(tmp_path):
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2]])
    path = tmp_path / "tri.ply"
    write_ply(path, V, F)
    data = path.read_bytes()
    header, body = data.split(b"end_header\n")
    assert b"format binary_little_endian 1.0" in header
    assert b"property float x" in header
    assert b"property list uchar int vertex_indices" in header
    assert len(body) == 3 * 3 * 4 + (1 + 3 * 4)
    npt.assert_allclose(np.frombuffer(body[:36], "<f4").reshape(3, 3), V, atol=0)


@pytest.mark.parametrize("fmt", ["binary_little_endian", "binary_big_endian", "ascii"])
def test_read_spheremap_layout(tmp_path, fmt):
    X, P, F = _random_mesh()
    path = tmp_path / f"m_{fmt}.ply"
    _write_spheremap_like(path, X, P, F, fmt)
    X2, P2, F2 = read_spheremap_ply(path)
    assert X2.dtype == np.float64 and P2.dtype == np.float64 and F2.dtype == np.int64
    npt.assert_allclose(X2, X, rtol=1e-6, atol=1e-6)  # float32 storage
    npt.assert_allclose(P2, P, rtol=1e-12, atol=1e-12)
    npt.assert_array_equal(F2, F)


def test_read_uses_header_property_order(tmp_path):
    # Properties in a different order and with sized type names.
    X, P, F = _random_mesh(1)
    n = len(X)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float64 pz\nproperty float32 x\nproperty float64 px\n"
        "property float32 y\nproperty float64 py\nproperty float32 z\n"
        f"element face {len(F)}\nproperty list uint8 uint32 vertex_indices\nend_header\n"
    )
    v = np.empty(n, dtype=[("pz", "<f8"), ("x", "<f4"), ("px", "<f8"), ("y", "<f4"), ("py", "<f8"), ("z", "<f4")])
    for k, a in zip("xyz", X.T):
        v[k] = a
    for k, a in zip(("px", "py", "pz"), P.T):
        v[k] = a
    fc = np.empty(len(F), dtype=[("n", "u1"), ("i", "<u4", (3,))])
    fc["n"], fc["i"] = 3, F
    path = tmp_path / "reordered.ply"
    path.write_bytes(header.encode() + v.tobytes() + fc.tobytes())
    X2, P2, F2 = read_spheremap_ply(path)
    npt.assert_allclose(X2, X, rtol=1e-6, atol=1e-6)
    npt.assert_allclose(P2, P, rtol=1e-12, atol=1e-12)
    npt.assert_array_equal(F2, F)


def test_read_rejects_plain_mesh_and_nontriangles(tmp_path):
    X, _, F = _random_mesh()
    plain = tmp_path / "plain.ply"
    write_ply(plain, X, F)
    with pytest.raises(ValueError, match="px"):
        read_spheremap_ply(plain)
    quad = tmp_path / "quad.ply"
    quad.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty double x\nproperty double y\n"
        b"property double z\nproperty double px\nproperty double py\nproperty double pz\n"
        b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        + np.zeros((4, 6), "<f8").tobytes()
        + np.array([4], "u1").tobytes()
        + np.arange(4, dtype="<i4").tobytes()
    )
    with pytest.raises(ValueError):
        read_spheremap_ply(quad)


def test_write_ply_rejects_bad_input(tmp_path):
    V = np.zeros((3, 3))
    with pytest.raises(ValueError):
        write_ply(tmp_path / "a.ply", V[:, :2], [[0, 1, 2]])
    with pytest.raises(ValueError):
        write_ply(tmp_path / "b.ply", V, [[0, 1, 3]])
    with pytest.raises(ValueError):
        write_ply(tmp_path / "c.ply", V, [[0, 1, 2]], coord_dtype="i4")


def test_write_read_empty_faces(tmp_path):
    path = tmp_path / "empty.ply"
    _write_spheremap_like(path, np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), np.zeros((0, 3), int))
    X, P, F = read_spheremap_ply(path)
    assert X.shape == (1, 3) and P.shape == (1, 3) and F.shape == (0, 3)


@pytest.mark.skipif(not SPHEREMAP.exists(), reason="SphereMap not built (scripts/build_mreg.sh)")
def test_spheremap_icosphere(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    V = np.array(mesh.vertices) * [2.0, 1.0, 1.0]
    F = np.array(mesh.faces, dtype=np.int64)
    src, out = tmp_path / "in.ply", tmp_path / "out.ply"
    write_ply(src, V, F)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run(
        [str(SPHEREMAP), "--in", str(src), "--out", str(out), "--threads", "1"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "Failed to meet centering" not in r.stdout + r.stderr
    X, P, F2 = read_spheremap_ply(out)
    npt.assert_allclose(X, V.astype(np.float32), rtol=0, atol=1e-6)
    npt.assert_array_equal(F2, F)
    npt.assert_allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-9)
    assert np.all(np.linalg.det(P[F]) > 0)  # no flipped spherical triangles
    w = trimesh.triangles.area(V[F])
    w /= w.sum()
    c = P[F].sum(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    assert np.linalg.norm((w[:, None] * c).sum(axis=0)) < 1e-8
