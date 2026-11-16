"""SphereMap stage: conformal spherical parametrization with MoebiusRegistration's SphereMap.

SphereMap reads vertex positions as float (Include/Misha/Ply.h), so the mesh it actually maps
is float32(V). That mesh is passed on as ``V_sub`` (float64 holding float32 values) and later
stages must use it as the surface geometry.

SphereMap exits 0 on every failure, so success is decided from stderr and the output file, in this order:

0. "Invalid option" warning (a flag was ignored)            -> ``spheremap_bad_option``
1. ``[ERROR]`` on stderr                                     -> ``spheremap_error``
2. final "Failed to meet centering threshold after N iterations" warning *without* the
   ", trying golden-section search" suffix                   -> ``centering_failed``
3. output file missing                                       -> ``spheremap_no_output``
4. non-finite P                                              -> ``spheremap_nonfinite``

plus a timeout (``spheremap_timeout``) and a correspondence check: SphereMap writes the input
x,y,z back as float, so X must equal float32(V) bitwise and F must be unchanged.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from moe_data.io_ply import read_spheremap_ply, write_ply
from moe_data.stages.base import StageResult

NAME = "spheremap"

# --mesh 1 selects the triangle-mesh (cotangent Laplacian) code path:
# SphereMap.cpp:45 enum MESH_TYPE_TRIANGLE = 0, SphereMap.cpp:61 parameter "mesh" (1-based),
# SphereMap.cpp:1211 "MeshType.value -= 1", SphereMap.cpp:1225 default -> PlyReadTriangles into
# TriangleMesh (SphereMap.cpp:233). 2 = polygon mesh, 3 = implicitly triangulated polygon mesh.
_MESH_TYPE_TRIANGLE = 1

# The centering cut-off is NOT settable from the command line: CutOff("cutOff", 1e-10f) is
# declared at SphereMap.cpp:62 but missing from the params[] list at SphereMap.cpp:101, so
# "--cutOff x" only prints "[WARNING] Invalid option" and the compiled default is used. The
# stage therefore refuses to run unless cfg spheremap.center_cutoff equals that default.
_COMPILED_CUTOFF = float(np.float32(1e-10))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FINAL_CENTERING = re.compile(r"Failed to meet centering threshold after \d+ iterations\s*$")


def resolve_binary(cfg: dict) -> Path:
    """cfg paths.spheremap_bin; a relative path is tried from the cwd, then the repo root."""
    p = Path(cfg["paths"]["spheremap_bin"])
    if p.is_absolute():
        return p
    return p.resolve() if p.exists() else _REPO_ROOT / p


def classify_stderr(stderr: str) -> str:
    """Return "" or the reject code implied by SphereMap's stderr (rules 1-2 above)."""
    lines = stderr.splitlines()
    if any("Invalid option" in ln or "should be of the form --<name>" in ln for ln in lines):
        return "spheremap_bad_option"   # a flag we passed was ignored by the binary
    if any("[ERROR]" in ln for ln in lines):
        return "spheremap_error"
    if any("[WARNING]" in ln and _FINAL_CENTERING.search(ln) for ln in lines):
        return "centering_failed"
    return ""


def run_spheremap(V_sub, F, cfg: dict, workdir) -> tuple[StageResult, dict]:
    """Run SphereMap on (V_sub, F) inside workdir (in.ply -> out.ply).

    Any existing out.ply is deleted first, so a stale file can never be read back.
    Returns (result, extra_data) where extra_data has X, P on success.
    """
    workdir = Path(workdir)
    inp, out = workdir / "in.ply", workdir / "out.ply"
    out.unlink(missing_ok=True)
    write_ply(inp, V_sub, F, coord_dtype="f4")

    sc = cfg["spheremap"]
    cmd = [str(resolve_binary(cfg)), "--in", str(inp), "--out", str(out),
           "--iters", str(int(sc["iters"])), "--stepSize", repr(float(sc["step_size"])),
           "--threads", "1",
           "--mesh", str(_MESH_TYPE_TRIANGLE)]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    timeout = float(cfg["run"]["spheremap_timeout_s"])
    meta: dict = {"spheremap_flags": " ".join(cmd[5:])}

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=workdir)
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, stderr = proc.communicate()
        meta.update(spheremap_runtime_s=time.perf_counter() - t0, spheremap_stderr=stderr or "",
                    spheremap_returncode=proc.returncode)
        return StageResult(False, meta=meta, reason=f"spheremap_timeout: > {timeout:g} s"), {}
    meta.update(spheremap_runtime_s=time.perf_counter() - t0, spheremap_stderr=stderr,
                spheremap_returncode=proc.returncode)

    code = classify_stderr(stderr)
    if code:
        last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), "")
        return StageResult(False, meta=meta, reason=f"{code}: {last.strip()}"), {}
    if not out.exists():
        return StageResult(False, meta=meta,
                           reason=f"spheremap_no_output: exit code {proc.returncode}, no output file"), {}
    try:
        X, P, F_out = read_spheremap_ply(out)
    except ValueError as e:
        return StageResult(False, meta=meta, reason=f"spheremap_no_output: unreadable output ({e})"), {}
    if not np.isfinite(P).all():
        n_bad = int((~np.isfinite(P).all(axis=1)).sum())
        return StageResult(False, meta=meta, reason=f"spheremap_nonfinite: {n_bad} vertices with non-finite P"), {}
    if X.shape != V_sub.shape or not np.array_equal(X, V_sub) or not np.array_equal(F_out, F):
        return StageResult(False, meta=meta,
                           reason="spheremap_correspondence: output x,y,z or faces differ from input"), {}
    return StageResult(True, meta=meta), {"X": X, "P": P}


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    V, F = inputs.get("V"), inputs.get("F")
    if V is None or F is None:
        return StageResult(False, reason="missing_input: V or F not in inputs")
    F = np.asarray(F, dtype=np.int64)
    V_sub = np.asarray(V, dtype=np.float64).astype(np.float32).astype(np.float64)
    if float(np.float32(cfg["spheremap"]["center_cutoff"])) != _COMPILED_CUTOFF:
        return StageResult(False, reason=f"spheremap_config: center_cutoff "
                           f"{cfg['spheremap']['center_cutoff']} != compiled default 1e-10 (not a CLI option)")
    binary = resolve_binary(cfg)
    if not binary.exists():
        return StageResult(False, reason=f"spheremap_missing_binary: {binary}")

    workdir = tempfile.mkdtemp(prefix=f"spheremap_{shape_id}_")
    try:
        res, extra = run_spheremap(V_sub, F, cfg, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if not res.ok:
        return res
    res.data = {"V_sub": V_sub, "F": F, "P": extra["P"], "X": extra["X"]}
    return res
