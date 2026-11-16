import pytest

from moe_data.config import load_config
from moe_data.thingi import (is_redistributable, list_candidates, load_mesh,
                             metadata_topology_ok)

# every licence string present in the Thingi10K 1.5.0 metadata (9,995 files)
LICENCES = {
    "Creative Commons - Attribution - Share Alike": True,
    "Creative Commons - Attribution": True,
    "Creative Commons - Attribution - Non-Commercial": False,
    "Attribution - Non-Commercial - Share Alike": False,
    "Attribution - Non-Commercial - No Derivatives": False,
    "GNU - GPL": True,
    "Creative Commons - Public Domain Dedication": True,
    "Public Domain": True,
    "Creative Commons - Attribution - No Derivatives": False,
    "BSD License": True,
    "unknown_license": False,
    "GNU - LGPL": True,
}


@pytest.mark.parametrize("licence,expected", sorted(LICENCES.items()))
def test_licence_redistributable(licence, expected):
    assert is_redistributable(licence) is expected


def test_missing_licence_not_redistributable():
    assert is_redistributable(None) is False
    assert is_redistributable("") is False


def test_metadata_topology_ok_codes():
    good = dict(closed=True, num_boundary_edges=0, edge_manifold=True, vertex_manifold=True,
                oriented=True, num_components=1, euler=2)
    assert metadata_topology_ok(good) == (True, "")
    assert metadata_topology_ok({**good, "euler": 0})[1].startswith("not_genus0: ")
    assert metadata_topology_ok({**good, "closed": False})[1].startswith("not_closed: ")
    prefixed = {f"meta_{k}": v for k, v in {**good, "num_components": 3}.items()}
    assert metadata_topology_ok(prefixed)[1].startswith("multiple_components: ")


@pytest.fixture(scope="module")
def cfg():
    return load_config("config.yaml")


@pytest.fixture(scope="module")
def all_candidates(cfg):
    return list_candidates(cfg, pilot=False)


def test_full_candidate_list(all_candidates):
    assert len(all_candidates) == 9995
    c = all_candidates[0]
    assert c["shape_id"] == f"thingi_{c['file_id']}"
    assert set(LICENCES) == {c["licence"] for c in all_candidates}
    for k in ("thing_id", "author", "file_path", "redistributable", "self_intersecting",
              "meta_euler", "meta_closed"):
        assert k in c


def test_pilot_deterministic_and_unfiltered(cfg, all_candidates):
    a = list_candidates(cfg, pilot=True)
    b = list_candidates(cfg, pilot=True)
    assert [c["shape_id"] for c in a] == [c["shape_id"] for c in b]
    assert len(a) == cfg["run"]["pilot_thingi"]
    other = list_candidates({**cfg, "seed": cfg["seed"] + 1}, pilot=True)
    assert [c["shape_id"] for c in a] != [c["shape_id"] for c in other]
    ids = {c["shape_id"] for c in all_candidates}
    assert {c["shape_id"] for c in a} <= ids


def test_load_mesh_dtypes(all_candidates):
    V, F = load_mesh(all_candidates[0])
    assert V.dtype.name == "float64" and F.dtype.name == "int64"
    assert V.shape == (all_candidates[0]["meta_num_vertices"], 3)
