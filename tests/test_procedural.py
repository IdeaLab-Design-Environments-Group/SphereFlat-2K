import copy
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from moe_data.config import load_config
from moe_data.procedural import generate, iter_specs, mesh_protrusion

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
N_SAMPLE = 10


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config(CONFIG))
    c["procedural"]["sdf_resolution"] = 48  # fast; thin capsules still forced to >= 3 voxels across
    c["procedural"]["blob"]["icosphere_subdivisions"] = 3
    return c


@pytest.fixture(scope="module")
def samples(cfg):
    return {fam: [generate(fam, i, cfg) for i in range(N_SAMPLE)] for fam in ("blob", "protrusion")}


def _edges(F):
    return np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])


def _n_components(n, F):
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in _edges(F):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(v) for v in np.unique(F)})


def _signed_volume(V, F):
    return np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0


# ----------------------------------------------------------------------------- specs

def test_iter_specs_pilot_split(cfg):
    specs = iter_specs(cfg, pilot=True)
    assert len(specs) == cfg["run"]["pilot_procedural"]
    fams = [f for _, f, _ in specs]
    assert fams.count("blob") == fams.count("protrusion") == 25
    assert ("proc_blob_0", "blob", 0) in specs and ("proc_prot_24", "protrusion", 24) in specs
    assert len({s for s, _, _ in specs}) == len(specs)


def test_iter_specs_full(cfg):
    specs = iter_specs(cfg, pilot=False)
    assert len(specs) == cfg["procedural"]["n_blobs"] + cfg["procedural"]["n_protrusion"]


def test_unknown_family_raises(cfg):
    with pytest.raises(ValueError):
        generate("torus", 0, cfg)


# ----------------------------------------------------------------------------- determinism

@pytest.mark.parametrize("family", ["blob", "protrusion"])
def test_deterministic(cfg, samples, family):
    V0, F0, p0 = samples[family][0]
    V1, F1, p1 = generate(family, 0, cfg)
    assert_array_equal(V0, V1)
    assert_array_equal(F0, F1)
    assert p0 == p1
    V2, _, p2 = samples[family][1]
    assert p0 != p2
    assert V0.shape != V2.shape or not np.allclose(V0, V2)


def test_seed_changes_output(cfg):
    other = copy.deepcopy(cfg)
    other["seed"] = cfg["seed"] + 1
    assert generate("blob", 0, cfg)[2] != generate("blob", 0, other)[2]


# ----------------------------------------------------------------------------- mesh validity

@pytest.mark.parametrize("family", ["blob", "protrusion"])
def test_meshes_closed_genus0_outward(samples, family):
    for V, F, params in samples[family]:
        assert V.dtype == np.float64 and V.ndim == 2 and V.shape[1] == 3
        assert F.dtype == np.int64 and F.ndim == 2 and F.shape[1] == 3
        directed = _edges(F)
        # watertight + consistently oriented: each directed edge once, its reverse also present
        assert len(np.unique(directed, axis=0)) == len(directed)
        und = np.unique(np.sort(directed, axis=1), axis=0)
        assert 2 * len(und) == len(directed)
        assert len(np.unique(F)) == len(V)  # no unreferenced vertices
        assert _n_components(len(V), F) == 1
        assert len(V) - len(und) + len(F) == 2
        assert _signed_volume(V, F) > 0
        json.dumps(params)


def test_blob_params_in_range(cfg, samples):
    bc = cfg["procedural"]["blob"]
    for V, _, p in samples["blob"]:
        assert bc["sh_max_degree"][0] <= p["sh_max_degree"] <= bc["sh_max_degree"][1]
        assert bc["amplitude"][0] <= p["amplitude"] <= bc["amplitude"][1]
        r = np.linalg.norm(V, axis=1)
        assert r.min() >= 0.2 * r.mean() - 1e-12
        # RMS radial displacement equals the amplitude (Parseval over orthonormal real SH)
        c = np.array([c for _, _, c in p["sh_coeffs"]])
        assert_allclose(np.sqrt(np.sum(c ** 2) / (4 * np.pi)), p["amplitude"], rtol=1e-9)


def test_protrusion_params_in_range(cfg, samples):
    prc = cfg["procedural"]["protrusion"]
    for _, _, p in samples["protrusion"]:
        prots = p["protrusions"]
        assert prc["count"][0] <= len(prots) == p["count"] <= prc["count"][1]
        for q in prots:
            assert prc["length"][0] <= q["length"] <= prc["length"][1]
            assert prc["thickness"][0] <= q["thickness"] <= prc["thickness"][1]
            assert_allclose(np.linalg.norm(q["direction"]), 1.0, atol=1e-12)
        assert_allclose(p["max_length"], max(q["length"] for q in prots))
        assert p["base_sh_coeffs"]


# ----------------------------------------------------------------------------- protrusion length

def test_longer_capsule_reaches_further(cfg, samples):
    _, _, base = samples["protrusion"][0]
    extents = []
    for length in (0.5, 1.5, 2.5):
        params = copy.deepcopy(base)
        params["protrusions"] = [{"direction": [0.0, 0.0, 1.0], "length": length, "thickness": 0.2}]
        V, F = mesh_protrusion(params, cfg)
        extents.append(np.linalg.norm(V - V.mean(axis=0), axis=1).max())
    assert np.all(np.diff(extents) > 0.3)


def test_thin_capsule_stays_connected(cfg, samples):
    _, _, base = samples["protrusion"][0]
    params = copy.deepcopy(base)
    coarse = copy.deepcopy(cfg)
    coarse["procedural"]["sdf_resolution"] = 16  # voxel >> capsule diameter unless refined
    t_min = cfg["procedural"]["protrusion"]["thickness"][0]
    l_max = cfg["procedural"]["protrusion"]["length"][1]
    d = np.array([1.0, 0.37, 0.21]) / np.linalg.norm([1.0, 0.37, 0.21])
    params["protrusions"] = [{"direction": d.tolist(), "length": l_max, "thickness": t_min}]
    V, F = mesh_protrusion(params, coarse)
    # the thinnest, longest capsule survives as part of the single kept component
    assert (V @ d).max() > 1.0 + 0.9 * l_max


def test_protrusions_respect_length_over_radius_cap():
    """The generator caps each protrusion at length/thickness <= 24, beyond
    which multi-protrusion maps collapse below float64 resolution."""
    import copy
    from moe_data.config import load_config
    cfg = copy.deepcopy(load_config("config.yaml"))
    cfg["procedural"]["sdf_resolution"] = 40
    cap = cfg["procedural"]["protrusion"]["max_length_over_radius"]
    for i in range(12):
        _, _, params = generate("protrusion", i, cfg)
        assert all(p["length"] / p["thickness"] <= cap for p in params["protrusions"])
        assert params["max_length_over_radius"] == max(p["length"] / p["thickness"] for p in params["protrusions"])
