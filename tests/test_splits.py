import json

import pandas as pd
import pytest

from moe_data.splits import assign_splits, procedural_severity

CFG = {"seed": 1, "split": {"train": 0.8, "val": 0.1, "test": 0.1,
                           "holdout_length_min": 2.4, "holdout_by": "length", "holdout_lr_min": 16},
       "protrusion": {"range_threshold": 6.0}}


def _thingi(i, group, psi_range, ok=True):
    return {"id": i, "ok": ok, "dup_group": group, "source": "thingi10k", "psi.psi_range": psi_range}


def _proc(i, length, thickness):
    params = {"protrusions": [{"length": length, "thickness": thickness}]}
    return {"id": i, "ok": True, "dup_group": i, "source": "procedural", "load.family": "protrusion",
            "load.params_json": json.dumps(params)}


def test_group_follows_any_qualifying_member():
    """A-B share a thing, B-C duplicates (one group); only C is a protrusion. Order must not matter."""
    rows = [_thingi("A", "G", 1.0), _thingi("B", "G", 2.0), _thingi("C", "G", 9.0)]
    for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
        df = pd.DataFrame([rows[i] for i in order])
        s = assign_splits(df, CFG)
        assert set(s) == {"protrusion_test"}


def test_groups_never_span_splits_and_fractions():
    rows = [_thingi(f"t{i}", f"g{i // 3}", 1.0) for i in range(3000)]
    df = pd.DataFrame(rows)
    df["split"] = assign_splits(df, CFG)
    assert df.groupby("dup_group")["split"].nunique().max() == 1
    frac = df.drop_duplicates("dup_group")["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.8, abs=0.04)
    assert frac["val"] == pytest.approx(0.1, abs=0.03)


def test_split_stable_when_other_groups_change():
    rows = [_thingi(f"t{i}", f"g{i}", 1.0) for i in range(200)]
    s1 = assign_splits(pd.DataFrame(rows), CFG)
    rows[0]["psi.psi_range"] = 9.0                      # one group becomes protrusion_test
    s2 = assign_splits(pd.DataFrame(rows), CFG)
    assert s2.iloc[0] == "protrusion_test"
    assert (s1.iloc[1:] == s2.iloc[1:]).all()


def test_rejected_shapes_get_no_split():
    df = pd.DataFrame([_thingi("A", "A", 1.0, ok=False), _thingi("B", "B", 1.0)])
    s = assign_splits(df, CFG)
    assert s.iloc[0] == "" and s.iloc[1] in ("train", "val", "test")


def test_procedural_holdout_rules():
    df = pd.DataFrame([_proc("p_long_thick", 2.5, 0.4), _proc("p_short_thin", 1.0, 0.05)])
    by_len = assign_splits(df, CFG)
    assert by_len.iloc[0] == "protrusion_test" and by_len.iloc[1] != "protrusion_test"
    cfg_lr = json.loads(json.dumps(CFG))
    cfg_lr["split"]["holdout_by"] = "length_over_radius"
    by_lr = assign_splits(df, cfg_lr)                  # L/r 6.25 vs 20
    assert by_lr.iloc[0] != "protrusion_test" and by_lr.iloc[1] == "protrusion_test"
    assert procedural_severity(df.iloc[1]["load.params_json"]) == pytest.approx((1.0, 20.0))


def test_rejected_qualifying_member_still_holds_out_its_group():
    """A member that qualifies (psi range 8) but is rejected downstream must still send
    its accepted group-mates to protrusion_test."""
    df = pd.DataFrame([_thingi("A", "G", 8.0, ok=False), _thingi("B", "G", 2.0)])
    s = assign_splits(df, CFG)
    assert s.iloc[0] == "" and s.iloc[1] == "protrusion_test"


def _cfg_new():
    cfg = json.loads(json.dumps(CFG))
    cfg["split"].update(holdout_by="length_over_radius", protrusion_inrange_fraction=0.15,
                        min_test_per_source={"thingi10k": 100})
    return cfg


def test_inrange_protrusion_test_set_and_holdout_are_disjoint():
    rows = [_proc(f"p{i}", 1.0 + (i % 10) * 0.2, 0.1) for i in range(400)]      # L/r 10 .. 28
    df = pd.DataFrame(rows)
    s = assign_splits(df, _cfg_new())
    lr = df["load.params_json"].map(lambda p: procedural_severity(p)[1])
    assert (s[lr > 16] == "protrusion_test").all()
    inr = s[lr <= 16]
    assert set(inr) <= {"train", "val", "test", "protrusion_test_inrange"}
    assert (inr == "protrusion_test_inrange").mean() == pytest.approx(0.15, abs=0.06)


def test_min_thingi_test_count_tops_up_with_whole_groups():
    rows = [_thingi(f"t{i}", f"g{i // 2}", 1.0) for i in range(600)]               # 300 groups of 2
    df = pd.DataFrame(rows)
    df["split"] = assign_splits(df, _cfg_new())
    assert (df["split"] == "test").sum() >= 100
    assert df.groupby("dup_group")["split"].nunique().max() == 1
    assert (df["split"] == "test").sum() <= 101                                    # stops once met


def test_min_test_keeps_hash_assignment_when_already_met():
    rows = [_thingi(f"t{i}", f"g{i}", 1.0) for i in range(3000)]
    cfg = _cfg_new()
    a = assign_splits(pd.DataFrame(rows), cfg)
    cfg["split"]["min_test_per_source"] = {}
    b = assign_splits(pd.DataFrame(rows), cfg)
    assert (a == b).all()                                                          # ~300 test already
