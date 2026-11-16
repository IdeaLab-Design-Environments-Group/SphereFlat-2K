"""Procedural genus-0 shapes with controlled difficulty.

Two families:

* ``blob``: unit sphere displaced radially by random real spherical harmonics. The surface is
  star-shaped, so it is meshed directly from a subdivided icosphere (genus 0 by construction).
* ``protrusion``: a low-amplitude blob plus 1-4 radial capsules, combined with a polynomial
  smooth-min of signed distances and meshed with marching cubes.

``generate(family, index, cfg)`` is deterministic in ``(cfg['seed'], family, index)``. Meshes are
not normalised or remeshed here; later stages do that.
"""
import zlib

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import sph_harm_y
from skimage.measure import marching_cubes

FAMILIES = ("blob", "protrusion")
_ID_PREFIX = {"blob": "proc_blob", "protrusion": "proc_prot"}

# Mesh resolution of the blob icosphere (subdivision level 5 -> 10242 vertices). Not a threshold;
# overridable with cfg['procedural']['blob']['icosphere_subdivisions'].
_DEFAULT_ICO_SUBDIV = 5
# Rejection rule from the task spec: min radius must be >= this fraction of the mean radius.
_MIN_RADIUS_FRAC = 0.2
# Capsule diameter must span at least this many voxels; resolution is raised otherwise.
_MIN_VOXELS_ACROSS = 3.0
_MAX_ATTEMPTS = 50


# ----------------------------------------------------------------------------- seeding / specs

def _rng(cfg: dict, family: str, index: int) -> np.random.Generator:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    ss = np.random.SeedSequence([int(cfg["seed"]), zlib.crc32(family.encode()), int(index)])
    return np.random.default_rng(ss)


def iter_specs(cfg: dict, pilot: bool) -> list[tuple[str, str, int]]:
    """(shape_id, family, index) for every procedural shape; pilot splits run.pilot_procedural evenly."""
    pc = cfg["procedural"]
    if pilot:
        total = int(cfg["run"]["pilot_procedural"])
        counts = {"blob": total - total // 2, "protrusion": total // 2}
    else:
        counts = {"blob": int(pc["n_blobs"]), "protrusion": int(pc["n_protrusion"])}
    return [(f"{_ID_PREFIX[fam]}_{k}", fam, k) for fam in FAMILIES for k in range(counts[fam])]


# ----------------------------------------------------------------------------- spherical harmonics

def _real_sh(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Orthonormal real spherical harmonic Y_lm (theta polar, phi azimuth in [0, 2pi])."""
    if m == 0:
        return sph_harm_y(l, 0, theta, phi).real
    y = sph_harm_y(l, abs(m), theta, phi)
    sign = (-1.0) ** m  # undo the Condon-Shortley phase
    return np.sqrt(2.0) * sign * (y.real if m > 0 else y.imag)


def _sph_angles(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    theta = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2.0 * np.pi)
    return theta, phi


def blob_radius(coeffs: list, dirs: np.ndarray) -> np.ndarray:
    """Radius function r(u) = 1 + sum c_lm Y_lm(u) at unit directions ``dirs`` (k,3).

    ``coeffs`` is a list of ``[l, m, c]`` with l >= 1, so the mean radius over the sphere is 1.
    """
    theta, phi = _sph_angles(np.asarray(dirs, dtype=np.float64))
    r = np.ones(len(theta))
    for l, m, c in coeffs:
        r += c * _real_sh(int(l), int(m), theta, phi)
    return r


def _sample_sh_coeffs(rng: np.random.Generator, l_max: int, amplitude: float,
                      check_dirs: np.ndarray) -> list:
    """Random real-SH coefficients with RMS radial displacement ``amplitude``; resamples until
    min r >= _MIN_RADIUS_FRAC * mean r (mean r is 1 because there is no l=0 term)."""
    lm = [(l, m) for l in range(1, l_max + 1) for m in range(-l, l + 1)]
    for _ in range(_MAX_ATTEMPTS):
        # Variance falls off as 1/l^2 so low degrees dominate (smoother surfaces).
        c = rng.standard_normal(len(lm)) / np.array([l for l, _ in lm], dtype=np.float64)
        # For orthonormal Y_lm, RMS over the sphere of sum c Y = sqrt(sum c^2 / 4pi).
        c *= amplitude / np.sqrt(np.sum(c ** 2) / (4.0 * np.pi))
        coeffs = [[int(l), int(m), float(ci)] for (l, m), ci in zip(lm, c)]
        if blob_radius(coeffs, check_dirs).min() >= _MIN_RADIUS_FRAC:
            return coeffs
    raise RuntimeError("could not sample a positive radius function")


def _ico_subdiv(cfg: dict) -> int:
    return int(cfg["procedural"].get("blob", {}).get("icosphere_subdivisions", _DEFAULT_ICO_SUBDIV))


# ----------------------------------------------------------------------------- mesh helpers

def _signed_volume(V: np.ndarray, F: np.ndarray) -> float:
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


def _largest_component(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(V)
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    adj = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    used = np.zeros(n, dtype=bool)
    used[F.ravel()] = True
    counts = np.bincount(labels[used])
    keep_label = np.argmax(counts)
    face_keep = labels[F[:, 0]] == keep_label
    F = F[face_keep]
    vert_keep = np.zeros(n, dtype=bool)
    vert_keep[F.ravel()] = True
    remap = -np.ones(n, dtype=np.int64)
    remap[vert_keep] = np.arange(vert_keep.sum())
    return V[vert_keep], remap[F]


def _is_closed_genus0(V: np.ndarray, F: np.ndarray) -> bool:
    """Every directed edge appears once and its reverse once; one component; chi = 2."""
    if len(F) == 0:
        return False
    if np.any(F[:, 0] == F[:, 1]) or np.any(F[:, 1] == F[:, 2]) or np.any(F[:, 0] == F[:, 2]):
        return False
    directed = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    if len(np.unique(directed, axis=0)) != len(directed):
        return False
    und = np.unique(np.sort(directed, axis=1), axis=0)
    if 2 * len(und) != len(directed):
        return False
    n_comp, _ = connected_components(
        coo_matrix((np.ones(len(und)), (und[:, 0], und[:, 1])), shape=(len(V), len(V))),
        directed=False)
    return n_comp == 1 and len(V) - len(und) + len(F) == 2


# ----------------------------------------------------------------------------- blob

def mesh_blob(coeffs: list, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    ico = trimesh.creation.icosphere(subdivisions=_ico_subdiv(cfg))
    U = np.asarray(ico.vertices, dtype=np.float64)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = U * blob_radius(coeffs, U)[:, None]
    F = np.asarray(ico.faces, dtype=np.int64)
    if _signed_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


def _generate_blob(rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    bc = cfg["procedural"]["blob"]
    l_max = int(rng.integers(bc["sh_max_degree"][0], bc["sh_max_degree"][1] + 1))
    amplitude = float(rng.uniform(*bc["amplitude"]))
    check = np.asarray(trimesh.creation.icosphere(subdivisions=_ico_subdiv(cfg)).vertices)
    coeffs = _sample_sh_coeffs(rng, l_max, amplitude, check)
    V, F = mesh_blob(coeffs, cfg)
    params = {"family": "blob", "sh_max_degree": l_max, "amplitude": amplitude,
              "sh_basis": "real_orthonormal_no_condon_shortley", "sh_coeffs": coeffs}
    return V, F, params


# ----------------------------------------------------------------------------- protrusion

def _smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth-min (Quilez): h = clip(0.5 + 0.5 (b - a)/k, 0, 1);
    min = lerp(b, a, h) - k h (1 - h)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def _capsule_sdf(P: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    ab = b - a
    t = np.clip((P - a) @ ab / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - a - t[:, None] * ab, axis=1) - radius


def _capsule_endpoints(coeffs: list, prot: dict) -> tuple[np.ndarray, np.ndarray]:
    """Capsule axis from the origin to (r_base(d) + length) * d: it sticks out `length` past the base."""
    d = np.asarray(prot["direction"], dtype=np.float64)
    r = blob_radius(coeffs, d[None, :])[0]
    return np.zeros(3), (r + prot["length"]) * d


def protrusion_sdf(P: np.ndarray, params: dict, smooth_k: float) -> np.ndarray:
    """Approximate signed distance (negative inside) of base blob smooth-unioned with capsules."""
    coeffs = params["base_sh_coeffs"]
    rho = np.linalg.norm(P, axis=1)
    dirs = np.where(rho[:, None] > 0, P, np.array([0.0, 0.0, 1.0]))
    d = rho - blob_radius(coeffs, dirs)
    for prot in params["protrusions"]:
        a, b = _capsule_endpoints(coeffs, prot)
        d = _smooth_min(d, _capsule_sdf(P, a, b, prot["thickness"]), smooth_k)
    return d


def _grid_spec(params: dict, cfg: dict) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    """Axis-aligned grid around the shape. ``sdf_resolution`` cells span the longest axis; the voxel
    is shrunk further so the thinnest capsule diameter covers >= _MIN_VOXELS_ACROSS voxels."""
    pc = cfg["procedural"]
    coeffs = params["base_sh_coeffs"]
    check = np.asarray(trimesh.creation.icosphere(subdivisions=3).vertices)
    r_max = blob_radius(coeffs, check).max() * 1.05
    pts = [np.full(3, -r_max), np.full(3, r_max)]
    for prot in params["protrusions"]:
        _, b = _capsule_endpoints(coeffs, prot)
        pts += [b - prot["thickness"], b + prot["thickness"]]
    pts = np.array(pts)
    margin = 2.0 * pc["protrusion"]["smooth_k"]
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    voxel = (hi - lo).max() / int(pc["sdf_resolution"])
    t_min = min(p["thickness"] for p in params["protrusions"])
    voxel = min(voxel, 2.0 * t_min / _MIN_VOXELS_ACROSS)
    lo -= 2 * voxel  # at least two outside layers so the isosurface is closed
    shape = tuple(int(np.ceil(s)) + 1 for s in (hi + 2 * voxel - lo) / voxel)
    return lo, voxel, shape


def mesh_protrusion(params: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Marching-cubes mesh (largest component, outward) of the protrusion shape described by params."""
    k = float(cfg["procedural"]["protrusion"]["smooth_k"])
    lo, voxel, shape = _grid_spec(params, cfg)
    axes = [lo[i] + voxel * np.arange(shape[i]) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    sdf = protrusion_sdf(G, params, k).reshape(shape)
    # Keep grid samples off the isovalue so marching cubes never creates coincident vertices.
    sdf[np.abs(sdf) < 1e-9 * voxel] = 1e-9 * voxel
    verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(voxel,) * 3, allow_degenerate=False)
    V = verts.astype(np.float64) + lo
    F = faces.astype(np.int64)
    V, F = _largest_component(V, F)
    if _signed_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def _sample_protrusion(rng: np.random.Generator, prc: dict) -> dict:
    """One protrusion: direction uniform, (length, thickness) uniform on the configured box, rejected while
    length/thickness exceeds ``max_length_over_radius`` (maps beyond ~25 collapse below float64)."""
    direction = [float(x) for x in _random_unit(rng)]
    cap = float(prc.get("max_length_over_radius", np.inf))
    while True:
        length, thickness = float(rng.uniform(*prc["length"])), float(rng.uniform(*prc["thickness"]))
        if length / thickness <= cap:
            return {"direction": direction, "length": length, "thickness": thickness}


def _generate_protrusion(rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    pc = cfg["procedural"]
    bc, prc = pc["blob"], pc["protrusion"]
    check = np.asarray(trimesh.creation.icosphere(subdivisions=4).vertices)
    for _ in range(_MAX_ATTEMPTS):
        l_max = int(rng.integers(bc["sh_max_degree"][0], bc["sh_max_degree"][1] + 1))
        base_amp = float(bc["amplitude"][0])  # "low amplitude" base = the blob range minimum
        coeffs = _sample_sh_coeffs(rng, l_max, base_amp, check)
        n = int(rng.integers(prc["count"][0], prc["count"][1] + 1))
        prots = [_sample_protrusion(rng, prc) for _ in range(n)]
        params = {"family": "protrusion", "base_sh_max_degree": l_max, "base_amplitude": base_amp,
                  "sh_basis": "real_orthonormal_no_condon_shortley", "base_sh_coeffs": coeffs,
                  "smooth_k": float(prc["smooth_k"]), "protrusions": prots,
                  "count": n, "max_length": max(p["length"] for p in prots),
                  "max_length_over_radius": max(p["length"] / p["thickness"] for p in prots)}
        V, F = mesh_protrusion(params, cfg)
        if _is_closed_genus0(V, F):
            return V, F, params
    raise RuntimeError("could not generate a closed genus-0 protrusion shape")


# ----------------------------------------------------------------------------- public entry point

def generate(family: str, index: int, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deterministic (V float64 (n,3), F int64 (m,3) CCW from outside, JSON-able params)."""
    rng = _rng(cfg, family, index)
    if family == "blob":
        return _generate_blob(rng, cfg)
    return _generate_protrusion(rng, cfg)
