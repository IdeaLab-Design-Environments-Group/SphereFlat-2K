from moe_data.run import _dedup

CFG = {"dedup": {"near_threshold": 0.02}}


def _row(i, source, desc, h, thing=None, nf=100):
    return {"id": i, "source": source, "_descriptor": desc, "_hash": h, "thing_id": thing, "_n_faces": nf}


def test_procedural_shapes_never_near_grouped():
    rows = [_row("proc_prot_0", "procedural", [0.0, 0.0], "h0"),
            _row("proc_prot_1", "procedural", [0.01, 0.0], "h1")]      # distance 0.01 < threshold
    recs = _dedup(rows, CFG)
    assert recs["proc_prot_0"]["keep"] and recs["proc_prot_1"]["keep"]
    assert recs["proc_prot_0"]["group_id"] != recs["proc_prot_1"]["group_id"]


def test_thingi_near_duplicates_grouped_and_one_kept():
    rows = [_row("thingi_1", "thingi10k", [0.0, 0.0], "a", thing=1, nf=50),
            _row("thingi_2", "thingi10k", [0.01, 0.0], "b", thing=2, nf=80)]
    recs = _dedup(rows, CFG)
    assert recs["thingi_1"]["group_id"] == recs["thingi_2"]["group_id"]
    assert recs["thingi_2"]["keep"] and not recs["thingi_1"]["keep"]     # more faces kept
    assert recs["thingi_1"]["reject_reason"] == "duplicate_of: thingi_2"


def test_exact_duplicate_across_sources_rejected():
    rows = [_row("thingi_5", "thingi10k", [0.0, 0.0], "same", thing=5),
            _row("proc_blob_0", "procedural", [1.0, 0.0], "same")]
    recs = _dedup(rows, CFG)
    assert sum(not r["keep"] for r in recs.values()) == 1
    assert recs["thingi_5"]["reject_reason"] == "duplicate_of: proc_blob_0"


def test_cross_source_exact_duplicate_merges_groups():
    """P (procedural) == A (Thingi) exactly; B shares A's thing. P, A and B must end up in
    one split group even though A is rejected."""
    rows = [_row("thingi_A", "thingi10k", [0.0, 0.0], "same", thing=7),
            _row("thingi_B", "thingi10k", [5.0, 0.0], "other", thing=7),
            _row("proc_blob_0", "procedural", [9.0, 0.0], "same")]
    recs = _dedup(rows, CFG)
    assert len({r["group_id"] for r in recs.values()}) == 1


def test_fallback_queues_order_members_after_representative():
    """If the representative fails phase B, the next member is tried."""
    from moe_data.run import _fallback_queues
    rows = [_row("thingi_1", "thingi10k", [0.0, 0.0], "a", thing=1, nf=50),
            _row("thingi_2", "thingi10k", [0.001, 0.0], "b", thing=2, nf=80),
            _row("thingi_3", "thingi10k", [0.002, 0.0], "c", thing=3, nf=60),
            _row("thingi_4", "thingi10k", [9.0, 0.0], "d", thing=4, nf=70)]
    recs = _dedup(rows, CFG)
    q = _fallback_queues(recs, rows)
    assert q == {"thingi_2": ["thingi_3", "thingi_1"]}       # by face count, then id; singletons absent


def test_flipped_rejections_split_into_folds_and_collapse():
    """A flipped-triangle rejection whose shortest sphere edge is below 1e-9 is compression at the float64
    limit (orientation is rounding noise), not a geometric fold."""
    import pandas as pd
    from moe_data.run import classify_sphere_rejections
    df = pd.DataFrame({"reject_stage": ["validate_sphere", "validate_sphere", "remesh", ""],
                       "reject_reason": ["flipped_triangles: 20", "flipped_triangles: 900", "remesh_hausdorff", ""],
                       "validate_sphere.min_sphere_edge": [1e-4, 1e-13, None, None]})
    out = classify_sphere_rejections(df, collapse_edge=1e-9)
    assert out["sphere_folded"].tolist() == [True, False, False, False]
    assert out["sphere_collapsed"].tolist() == [False, True, False, False]
