"""MoebiusSphereDataset, processed-file I/O, subsampling, collate and worker seeding.

Frame oracles use only what a sample returns (P, psi, grid) plus analytic fields; they never call the
rotation helpers, so a frame mismatch between grid and P cannot cancel out.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import trimesh

from moe_data import geometry as geo
from moe_data import grid as g
from moe_data.config import load_config
from moe_data.dataset import (MoebiusSphereDataset, SourceBalancedSampler, area_weighted_subsample, collate,
                              make_loader, read_processed, source_groups, to_device, write_processed)
from moe_data.stages import grid as grid_stage

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
GH, GW = 64, 128
ABC = np.array([1.0, 0.7, 0.5])


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config(CONFIG))
    c["grid"].update(height=GH, width=GW, psi_source="psi")   # independent of config.yaml's choice
    return c


def _shape(sub, abc):
    """Star-shaped ellipsoid over an icosphere: P is the exact radial projection (all det > 0)."""
    m = trimesh.creation.icosphere(subdivisions=sub)
    P = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    F = np.asarray(m.faces, np.int64)
    V = P / np.sqrt(np.sum(np.square(P / abc), axis=1))[:, None]
    return V, F, P


def _arrays(V, F, P, psi, cfg, rng):
    area = geo.vertex_areas_barycentric(V, F)
    K_int = geo.angle_defect(V, F)
    d = {"V": V, "F": F, "P": P, "area": area, "area_sphere": geo.vertex_areas_barycentric(P, F), "psi": psi,
         "K_int": K_int, "K_pt": K_int / area, "H": geo.mean_curvature(V, F),
         "vis": rng.random((len(V), 2)).astype(np.float32)}
    res = grid_stage.run("x", d, cfg)
    assert res.ok, res.reason
    return {**d, **res.data}


@pytest.fixture(scope="module")
def processed(tmp_path_factory, cfg):
    root = tmp_path_factory.mktemp("processed")
    rng = np.random.default_rng(0)
    rows = []
    for k, (sub, abc) in enumerate([(4, ABC), (3, np.array([0.6, 1.0, 0.8])), (4, np.array([0.9, 0.5, 1.0]))]):
        V, F, P = _shape(sub, abc)
        write_processed(root / f"s{k}.npz", _arrays(V, F, P, P[:, 2].copy(), cfg, rng))   # psi = z (oracle)
        rows.append({"id": f"s{k}", "split": "train"})
    rows.append({"id": "rejected", "split": ""})
    return root, pd.DataFrame(rows)


# ------------------------------------------------------------------ I/O


def test_write_read_roundtrip_and_dtypes(processed, tmp_path):
    root, _ = processed
    d = read_processed(root / "s0.npz")
    assert d["P"].dtype == np.float64 and d["F"].dtype == np.int64
    assert d["vis"].dtype == np.float32 and d["grid"].dtype == np.float32
    assert list(d["grid_channels"]) == list(g.CHANNELS)
    with pytest.raises(KeyError):
        write_processed(tmp_path / "bad.npz", {k: v for k, v in d.items() if k != "vis"})
    with pytest.raises(ValueError):
        write_processed(tmp_path / "bad.npz", {**d, "vis": d["vis"][:-1]})


def test_split_filter_and_unrotated_item(processed, cfg):
    root, df = processed
    ds = MoebiusSphereDataset(df, root, "train", cfg, rotate=False, seed=1)
    assert ds.ids == ["s0", "s1", "s2"]
    it, d = ds[0], read_processed(root / "s0.npz")
    assert it["P"].dtype == torch.float64 and it["grid"].dtype == torch.float32
    np.testing.assert_array_equal(it["P"].numpy(), d["P"])
    np.testing.assert_array_equal(it["grid"].numpy(), d["grid"])
    np.testing.assert_array_equal(it["R"].numpy(), np.eye(3))


# ------------------------------------------------------------------ rotation frame


def test_rotated_grid_and_P_share_a_frame(processed, cfg):
    """psi_v = z of the ORIGINAL sphere position. In the returned frame psi_v = P'_v . d for an unknown
    direction d, recovered by least squares from (P, psi) alone. The grid's psi channel must then be the
    linear field c . d, its argmax must sit at d, and a cone there must hold the vertex maximum."""
    root, df = processed
    ds = MoebiusSphereDataset(df, root, "train", cfg, rotate=True, seed=3)
    C = g.latlong_cell_centers(GH, GW)
    cell = np.pi / GH
    for i in [0, 1, 2, 0]:
        it = ds[i]
        P, psi, G = it["P"].numpy(), it["psi"].numpy(), it["grid"].numpy()
        d, *_ = np.linalg.lstsq(P, psi, rcond=None)
        assert abs(np.linalg.norm(d) - 1) < 1e-12               # psi is exactly linear in P'
        assert np.linalg.norm(d - [0, 0, 1]) > 1e-3             # a rotation actually happened
        gpsi = G[0].ravel()
        np.testing.assert_allclose(gpsi, C @ d, atol=5e-3)
        c_star = C[np.argmax(gpsi)]
        assert np.arccos(np.clip(c_star @ d, -1, 1)) < 1.5 * cell
        ang = np.arccos(np.clip(P @ c_star, -1, 1))
        cone = np.clip(1 - ang / 0.3, 0, None)                  # cone of radius 0.3 rad at the argmax
        assert np.sum(cone * psi) / np.sum(cone) > np.cos(0.3)
        assert np.argmax(psi) in np.flatnonzero(ang < 0.2)


def test_rotated_density_moments_match_rotated_vertices(processed, cfg):
    """K_int_density of the rotated sample has the moments of K_int placed at the returned P."""
    root, df = processed
    ds = MoebiusSphereDataset(df, root, "train", cfg, rotate=True, seed=4)
    it = ds[0]
    K_int = read_processed(root / "s0.npz")["K_int"][it["index"].numpy()]
    P, G = it["P"].numpy(), it["grid"].numpy()[list(g.CHANNELS).index("K_int_density")].ravel()
    C = g.latlong_cell_centers(GH, GW)
    omega = np.repeat(g.latlong_cell_solid_angle(GH, GW), GW)
    for h in (lambda X: X[:, 0], lambda X: X[:, 1], lambda X: X[:, 2], lambda X: X[:, 0] * X[:, 1]):
        err = abs(np.sum(G * h(C) * omega) - np.sum(K_int * h(P))) / np.sum(np.abs(K_int))
        assert err < 5e-3


def test_rotations_are_proper_and_differ(processed, cfg):
    root, df = processed
    ds = MoebiusSphereDataset(df, root, "train", cfg, rotate=True, seed=5)
    Rs = [ds[i % 3]["R"].numpy() for i in range(6)]
    for R in Rs:
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert len({R.tobytes() for R in Rs}) == 6


# ------------------------------------------------------------------ subsampling


def test_area_weighted_subsample_is_unbiased():
    rng = np.random.default_rng(6)
    area = rng.gamma(0.5, size=400)                          # very uneven weights
    f = rng.normal(size=400) + 3 * (area > np.median(area))
    exact = np.sum(area * f)
    est = []
    for _ in range(4000):
        idx, w = area_weighted_subsample(area, 50, rng)
        assert np.allclose(w, area.sum() / 50)
        est.append(np.sum(w * f[idx]))
    est = np.array(est)
    se = est.std(ddof=1) / np.sqrt(len(est))
    assert abs(est.mean() - exact) < 4 * se
    # a wrong (uniform) sampler with the same weights is biased by many standard errors
    biased = [np.sum(area.sum() / 50 * f[rng.integers(0, 400, 50)]) for _ in range(4000)]
    assert abs(np.mean(biased) - exact) > 10 * se


def test_dataset_subsample_shapes(processed, cfg):
    root, df = processed
    it = MoebiusSphereDataset(df, root, "train", cfg, rotate=True, n_vertices=300, seed=7)[1]
    assert it["P"].shape == (300, 3) and it["vis"].shape == (300, 2) and it["psi"].shape == (300,)
    total = read_processed(root / "s1.npz")["area"].sum()
    assert float(it["area"].sum()) == pytest.approx(total, rel=1e-12)
    np.testing.assert_allclose(np.linalg.norm(it["P"].numpy(), axis=1), 1.0, atol=1e-14)


# ------------------------------------------------------------------ collate and device


def test_collate_pads_and_masks(processed, cfg):
    root, df = processed
    ds = MoebiusSphereDataset(df, root, "train", cfg, rotate=False, seed=8)
    items = [ds[0], ds[1]]                                    # 2562 and 642 vertices
    b = collate(items)
    n = [len(x["P"]) for x in items]
    assert b["grid"].shape == (2, 7, GH, GW) and b["P"].shape == (2, max(n), 3)
    assert b["mask"].dtype == torch.bool and b["mask"].sum(1).tolist() == n
    assert torch.equal(b["P"][1, :n[1]], items[1]["P"]) and torch.all(b["P"][1, n[1]:] == 0)
    assert torch.all(b["area"][~b["mask"]] == 0) and b["vis"].shape == (2, max(n), 2)
    assert b["id"] == ["s0", "s1"] and b["P"].dtype == torch.float64


def test_to_device_keeps_float64_on_cpu(processed, cfg):
    root, df = processed
    b = collate([MoebiusSphereDataset(df, root, "train", cfg, rotate=False)[1]])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = to_device(b, dev)
    assert out["grid"].device.type == dev and out["P"].device.type == "cpu" and out["P"].dtype == torch.float64
    cast = to_device(b, dev, float64="cast")
    assert cast["P"].device.type == dev and cast["P"].dtype == torch.float32
    with pytest.raises(ValueError):
        to_device(b, dev, float64="keep")


# ------------------------------------------------------------------ corner mask


@pytest.fixture
def cornered(processed, tmp_path):
    """s1 (642 vertices) with corner_dist = 0.001 * vertex id, s2 (2562) without corner_dist."""
    root, _ = processed
    d = read_processed(root / "s1.npz")
    d["corner_dist"] = 1e-3 * np.arange(len(d["P"]), dtype=np.float64)
    write_processed(tmp_path / "c1.npz", d)
    d2 = read_processed(root / "s2.npz")
    d2.pop("corner_dist", None)
    write_processed(tmp_path / "c2.npz", d2)
    return tmp_path, pd.DataFrame([{"id": "c1", "split": "train"}, {"id": "c2", "split": "train"}])


def test_corner_mask_matches_radius(cornered, cfg):
    root, df = cornered
    r = cfg["corners"]["exclude_radius"]
    it = MoebiusSphereDataset(df, root, "train", cfg, rotate=False)[0]
    cm = it["corner_mask"]
    assert cm.dtype == torch.bool and cm.shape == (642,)
    np.testing.assert_array_equal(cm.numpy(), 1e-3 * np.arange(642) < r)
    assert 0 < int(cm.sum()) < 642
    it = MoebiusSphereDataset(df, root, "train", cfg, rotate=False, corner_radius=0.1)[0]
    assert int(it["corner_mask"].sum()) == int(np.sum(1e-3 * np.arange(642) < 0.1))
    it = MoebiusSphereDataset(df, root, "train", cfg, rotate=False, corner_radius=0.0)[0]
    assert not it["corner_mask"].any()


def test_absent_corner_dist_gives_no_corners(cornered, cfg):
    root, df = cornered
    it = MoebiusSphereDataset(df, root, "train", cfg, rotate=False, corner_radius=10.0)[1]
    assert it["corner_mask"].shape == (2562,) and not it["corner_mask"].any()


def test_corner_mask_follows_subsampling_and_padding(cornered, cfg):
    root, df = cornered
    ds = MoebiusSphereDataset(df, root, "train", cfg, rotate=False, n_vertices=200, corner_radius=0.3, seed=2)
    it = ds[0]
    idx = it["index"].numpy()
    np.testing.assert_array_equal(it["corner_mask"].numpy(), 1e-3 * idx < 0.3)
    assert it["corner_mask"].any() and not it["corner_mask"].all()
    full = MoebiusSphereDataset(df, root, "train", cfg, rotate=False, corner_radius=0.3)
    a, b = full[0], full[1]                                  # 642 and 2562 vertices
    out = collate([a, b])
    assert out["corner_mask"].shape == (2, 2562) and out["corner_mask"].dtype == torch.bool
    assert torch.equal(out["corner_mask"][0, :642], a["corner_mask"])
    assert not out["corner_mask"][~out["mask"]].any() and out["mask"].sum(1).tolist() == [642, 2562]


# ------------------------------------------------------------------ source-balanced batches


def _manifest(counts, ids=("s0",)):
    """Train rows with the given (thingi10k, proc_blob, proc_protrusion) counts, ids cycling over ``ids``."""
    kinds = [("thingi10k", None), ("procedural", "blob"), ("procedural", "protrusion")]
    rows = [{"source": s, "load.family": f} for (s, f), c in zip(kinds, counts) for _ in range(c)]
    for k, r in enumerate(rows):
        r.update(id=ids[k % len(ids)], split="train")
    return pd.DataFrame(rows)


def test_source_groups_from_manifest(processed, cfg):
    root, _ = processed
    df = _manifest((2, 1, 1))
    assert source_groups(df) == ["thingi10k", "thingi10k", "proc_blob", "proc_protrusion"]
    assert source_groups(df[["id", "split"]]) == ["unknown"] * 4
    ds = MoebiusSphereDataset(df, root, "train", cfg, rotate=False)
    assert ds.groups == source_groups(df) and ds[2]["source_group"] == "proc_blob"


@pytest.mark.parametrize("weights,expect", [(None, (1 / 3, 1 / 3, 1 / 3)), ((2, 1, 1), (0.5, 0.25, 0.25)),
                                            ({"thingi10k": 2, "proc_blob": 1, "proc_protrusion": 1},
                                             (0.5, 0.25, 0.25))])
def test_balanced_sampler_batch_composition(processed, cfg, weights, expect):
    root, _ = processed
    ds = MoebiusSphereDataset(_manifest((90, 5, 5)), root, "train", cfg, rotate=False)
    s = SourceBalancedSampler(ds, batch_size=10, weights=weights, seed=0)
    assert s.group_names == ["thingi10k", "proc_blob", "proc_protrusion"] and len(s) == 10
    groups = np.array(ds.groups)
    tally = np.zeros(3)
    for epoch in range(50):
        s.set_epoch(epoch)
        batches = list(s)
        assert sum(map(len, batches)) == len(ds)
        for b in batches:
            assert len(b) == 10
            c = np.array([np.sum(groups[b] == g) for g in s.group_names])
            assert np.all(np.abs(c - 10 * np.array(expect)) < 1)   # every batch within one item of target
            tally += c
    np.testing.assert_allclose(tally / tally.sum(), expect, atol=0.01)


def test_balanced_sampler_partial_last_batch_and_bad_weights(processed, cfg):
    root, _ = processed
    ds = MoebiusSphereDataset(_manifest((90, 5, 5)), root, "train", cfg, rotate=False)
    s = SourceBalancedSampler(ds, batch_size=8)
    sizes = [len(b) for b in s]
    assert len(sizes) == 13 and sizes[-1] == 4 and sum(sizes) == 100
    for bad in [(1, 1), (1, -1, 1), (0, 0, 0)]:
        with pytest.raises(ValueError):
            SourceBalancedSampler(ds, 8, weights=bad)
    with pytest.raises(KeyError):
        SourceBalancedSampler(ds, 8, weights={"other": 1})


def test_balanced_sampler_deterministic_per_seed_and_epoch(processed, cfg):
    root, _ = processed
    ds = MoebiusSphereDataset(_manifest((90, 5, 5)), root, "train", cfg, rotate=False)
    a, b, c = (SourceBalancedSampler(ds, 10, seed=k) for k in (1, 1, 2))
    assert list(a) == list(b) and list(a) != list(c)
    e0 = list(a)
    a.set_epoch(1)
    b.set_epoch(1)
    assert list(a) == list(b) and list(a) != e0
    a.set_epoch(0)
    assert list(a) == e0


def test_balanced_loader_with_workers(processed, cfg):
    import multiprocessing
    import time
    root, _ = processed
    ds = MoebiusSphereDataset(_manifest((8, 2, 2), ids=("s1",)), root, "train", cfg, rotate=False)
    loader = make_loader(ds, batch_size=3, num_workers=2, seed=0, balance_by="source")
    t0 = time.monotonic()
    batches = list(loader)
    assert time.monotonic() - t0 < 4                          # no 5 s per-worker shutdown wait
    assert len(batches) == 4 and all(sorted(b["source_group"]) == ["proc_blob", "proc_protrusion", "thingi10k"]
                                     for b in batches)
    assert all(isinstance(b["corner_mask"], torch.Tensor) and b["corner_mask"].shape == (3, 642) for b in batches)
    assert not multiprocessing.active_children()              # workers exited
    with pytest.raises(ValueError):
        make_loader(ds, batch_size=3, balance_by="family")


# ------------------------------------------------------------------ worker seeding


def _rotations(ds, num_workers, seed):
    loader = make_loader(ds, batch_size=1, num_workers=num_workers, seed=seed, shuffle=False)
    return [b["R"][0].numpy() for b in loader]


def test_worker_seeding_differs_across_workers_and_is_reproducible(processed, cfg):
    root, df = processed
    df6 = pd.concat([df[df["split"] == "train"]] * 2, ignore_index=True)   # 6 items, 2 workers
    c = copy.deepcopy(cfg)
    c["grid"].update(height=16, width=32)
    ds = MoebiusSphereDataset(df6, root, "train", c, rotate=True, seed=9)
    a = _rotations(ds, 2, seed=11)
    b = _rotations(ds, 2, seed=11)
    other = _rotations(ds, 2, seed=12)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len({R.tobytes() for R in a}) == len(a)             # workers 0 and 1 drew different streams
    assert not np.array_equal(a[0], other[0])


def test_unrotated_item_refuses_grid_from_other_psi_estimator(processed, cfg):
    """A stored grid built from psi must not be paired with psi_lsq vertex values."""
    import copy
    root, df = processed
    cfg2 = copy.deepcopy(cfg)
    cfg2["grid"]["psi_source"] = "psi_lsq"
    d = read_processed(root / "s0.npz")
    if "psi_lsq" not in d:
        d["psi_lsq"] = d["psi"].copy()
    d["grid_psi_source"] = np.array("psi")
    write_processed(root / "s0.npz", d)
    ds = MoebiusSphereDataset(df, root, "train", cfg2, rotate=False, seed=1)
    with pytest.raises(ValueError, match="stored grid was built from 'psi'"):
        ds[0]
