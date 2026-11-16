import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import trimesh

from moe_data.config import load_config
from moe_data.stages import spheremap

CFG = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
HAVE_BIN = spheremap.resolve_binary(CFG).exists()
needs_bin = pytest.mark.skipif(not HAVE_BIN, reason="SphereMap binary not built")


def code(reason):
    return reason.split(":")[0]


def small_sphere(sub=3):
    m = trimesh.creation.icosphere(sub)
    V = m.vertices * 0.37 + 0.01  # off-centre, values not exactly representable in float32
    return V, m.faces.astype(np.int64)


def write_fake_output(path, X, P, F):
    """A PLY laid out like SphereMap's output (float x,y,z; double px,py,pz; uchar rgb)."""
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(X)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property double px\nproperty double py\nproperty double pz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    v = np.zeros(len(X), dtype=[("x", "<f4", 3), ("p", "<f8", 3), ("c", "u1", 3)])
    v["x"], v["p"] = X, P
    f = np.zeros(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)])
    f["n"], f["i"] = 3, F
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(v.tobytes())
        fh.write(f.tobytes())


def fake_binary(tmp_path, stderr_text="", ply=None, sleep=0.0):
    """Executable that mimics SphereMap: prints stderr_text, copies `ply` to --out, exits 0."""
    script = tmp_path / "FakeSphereMap"
    script.write_text(
        f"#!{sys.executable}\n"
        "import shutil, sys, time\n"
        f"time.sleep({sleep!r})\n"
        f"sys.stderr.write({stderr_text!r})\n"
        f"src = {str(ply) if ply else None!r}\n"
        "if src: shutil.copy(src, sys.argv[sys.argv.index('--out') + 1])\n")
    script.chmod(0o755)
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["spheremap_bin"] = str(script)
    return cfg


def sphere_P(V):
    c = V - V.mean(0)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


# ---------------------------------------------------------------- stderr classification

def test_classify_stderr_rules():
    gss = "[WARNING] Failed to meet centering threshold after 50 iterations, trying golden-section search\n"
    final = "[WARNING] Failed to meet centering threshold after 50 iterations\n"
    assert spheremap.classify_stderr("") == ""
    assert spheremap.classify_stderr(gss) == ""                 # GSS retry is informational
    assert spheremap.classify_stderr(gss + final) == "centering_failed"
    assert spheremap.classify_stderr("[ERROR] Failed to factorize matrix\n" + final) == "spheremap_error"


# ---------------------------------------------------------------- real binary

@needs_bin
def test_icosphere_maps_to_unit_sphere_with_exact_correspondence():
    V, F = small_sphere()
    res = spheremap.run("ico3", {"V": V, "F": F}, CFG)
    assert res.ok, res.reason
    d = res.data
    V32 = V.astype(np.float32)
    assert d["V_sub"].dtype == np.float64 and np.array_equal(d["V_sub"], V32.astype(np.float64))
    assert np.array_equal(d["X"], d["V_sub"])                   # bitwise correspondence
    assert np.array_equal(d["F"], F)
    assert d["P"].dtype == np.float64 and np.isfinite(d["P"]).all()
    npt.assert_allclose(np.linalg.norm(d["P"], axis=1), 1.0, atol=1e-9)
    assert res.meta["spheremap_runtime_s"] > 0
    assert res.meta["spheremap_stderr"] == ""
    assert "--mesh 1" in res.meta["spheremap_flags"] and "--threads 1" in res.meta["spheremap_flags"]


def _stale_workdir(tmp_path):
    """Workdir holding a valid SphereMap output from a successful run."""
    V, F = small_sphere(2)
    wd = tmp_path / "work"
    wd.mkdir()
    res, extra = spheremap.run_spheremap(V.astype(np.float32).astype(np.float64), F, CFG, wd)
    assert res.ok and (wd / "out.ply").exists()
    return wd


def _unreferenced():
    V, F = small_sphere(2)
    return np.vstack([V, [[5.0, 5.0, 5.0]]]), F


def _nan_coordinate():
    V, F = small_sphere(2)
    V = V.copy()
    V[0] = np.nan
    return V, F


def _two_components():
    V, F = small_sphere(2)
    return np.vstack([V, V + 3.0]), np.vstack([F, F + len(V)])


@needs_bin
@pytest.mark.parametrize("make, expected", [
    (_unreferenced, "spheremap_error"),       # Cholesky failure, no output
    (_nan_coordinate, "centering_failed"),    # final centering warning, NaN output
    (_two_components, "spheremap_error"),     # Cholesky failure, no output
])
def test_inv4_failures_reject_and_ignore_stale_output(tmp_path, make, expected):
    wd = _stale_workdir(tmp_path)
    V, F = make()
    res, extra = spheremap.run_spheremap(V.astype(np.float32).astype(np.float64), F, CFG, wd)
    assert not res.ok and code(res.reason) == expected, res.reason
    assert extra == {} and res.data == {}
    assert res.meta["spheremap_stderr"]


@needs_bin
def test_stage_run_rejects_failure(tmp_path):
    V, F = _two_components()
    res = spheremap.run("two", {"V": V, "F": F}, CFG)
    assert not res.ok and code(res.reason) == "spheremap_error"
    assert res.data == {}


# ---------------------------------------------------------------- fake binary (rules 3-4, timeout)

def test_missing_output_rejects(tmp_path):
    cfg = fake_binary(tmp_path)
    V, F = small_sphere(1)
    res = spheremap.run("x", {"V": V, "F": F}, cfg)
    assert not res.ok and code(res.reason) == "spheremap_no_output"


def test_stale_output_deleted_before_run(tmp_path):
    cfg = fake_binary(tmp_path)              # writes nothing
    V, F = small_sphere(1)
    V_sub = V.astype(np.float32).astype(np.float64)
    wd = tmp_path / "wd"
    wd.mkdir()
    write_fake_output(wd / "out.ply", V_sub, sphere_P(V_sub), F)
    res, extra = spheremap.run_spheremap(V_sub, F, cfg, wd)
    assert not res.ok and code(res.reason) == "spheremap_no_output"


def test_nonfinite_output_rejects(tmp_path):
    V, F = small_sphere(1)
    V_sub = V.astype(np.float32).astype(np.float64)
    P = sphere_P(V_sub)
    P[3] = np.nan
    write_fake_output(tmp_path / "out_src.ply", V_sub, P, F)
    cfg = fake_binary(tmp_path, ply=tmp_path / "out_src.ply")
    res = spheremap.run("x", {"V": V, "F": F}, cfg)
    assert not res.ok and code(res.reason) == "spheremap_nonfinite"


def test_gss_warning_alone_is_accepted(tmp_path):
    V, F = small_sphere(1)
    V_sub = V.astype(np.float32).astype(np.float64)
    write_fake_output(tmp_path / "out_src.ply", V_sub, sphere_P(V_sub), F)
    gss = "[WARNING] Failed to meet centering threshold after 50 iterations, trying golden-section search\n"
    cfg = fake_binary(tmp_path, stderr_text=gss, ply=tmp_path / "out_src.ply")
    res = spheremap.run("x", {"V": V, "F": F}, cfg)
    assert res.ok, res.reason
    assert res.meta["spheremap_stderr"] == gss


def test_correspondence_mismatch_rejects(tmp_path):
    V, F = small_sphere(1)
    V_sub = V.astype(np.float32).astype(np.float64)
    X = V_sub.copy()
    X[0, 0] += 1e-3
    write_fake_output(tmp_path / "out_src.ply", X, sphere_P(V_sub), F)
    cfg = fake_binary(tmp_path, ply=tmp_path / "out_src.ply")
    res = spheremap.run("x", {"V": V, "F": F}, cfg)
    assert not res.ok and code(res.reason) == "spheremap_correspondence"


def test_timeout_kills_and_rejects(tmp_path):
    cfg = fake_binary(tmp_path, sleep=30.0)
    cfg["run"]["spheremap_timeout_s"] = 0.5
    V, F = small_sphere(1)
    res = spheremap.run("x", {"V": V, "F": F}, cfg)
    assert not res.ok and code(res.reason) == "spheremap_timeout"
    assert res.meta["spheremap_runtime_s"] < 10


def test_missing_binary_rejects():
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["spheremap_bin"] = "/nonexistent/SphereMap"
    V, F = small_sphere(1)
    res = spheremap.run("x", {"V": V, "F": F}, cfg)
    assert not res.ok and code(res.reason) == "spheremap_missing_binary"


def test_ignored_flag_is_rejected():
    # SphereMap ignores unknown flags with only a warning (e.g. --cutOff is not a CLI option)
    err = "[WARNING] Invalid option: --cutOff\n[WARNING] Parameter name should be of the form --<name>: 1e-10\n"
    assert spheremap.classify_stderr(err) == "spheremap_bad_option"


def test_center_cutoff_other_than_compiled_default_rejects():
    cfg = copy.deepcopy(CFG)
    cfg["spheremap"]["center_cutoff"] = 1e-8
    V, F = small_sphere(1)
    res = spheremap.run("x", {"V": V, "F": F}, cfg)
    assert not res.ok and code(res.reason) == "spheremap_config"
