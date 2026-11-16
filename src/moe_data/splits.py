"""Train / val / test / protrusion_test assignment, by group, never by file.

Groups are the dedup groups (connected components of exact duplicates, near duplicates and shared
Thingi10K thing_id), so a group never spans two splits. A group goes to ``protrusion_test`` when ANY
member qualifies (a Thingi10K shape whose psi range exceeds the protrusion threshold, or a procedural
protrusion shape in the held-out severity range). Other groups are placed by a hash of
(seed, group id), so changing a threshold moves only the groups whose qualification changed.

Procedural protrusion groups below the hold-out range go, with probability ``protrusion_inrange_fraction``
(by the same hash), to ``protrusion_test_inrange``: an in-distribution protrusion test set to compare
against the out-of-range hold-out. ``min_test_per_source`` then moves whole train groups (in hash order)
to test until each listed source has at least that many test shapes.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

SPLITS = ("train", "val", "test")


def procedural_severity(params_json: str) -> tuple[float, float]:
    """(max protrusion length, max length/thickness) from the stored generator parameters."""
    p = json.loads(params_json)
    prots = p.get("protrusions") or []
    if not prots:
        return 0.0, 0.0
    return (max(float(q["length"]) for q in prots),
            max(float(q["length"]) / float(q["thickness"]) for q in prots))


def qualifies_protrusion_test(row: pd.Series, cfg: dict) -> bool:
    if row["source"] == "procedural":
        if row.get("load.family") != "protrusion" or not isinstance(row.get("load.params_json"), str):
            return False
        length, lr = procedural_severity(row["load.params_json"])
        rule = cfg["split"]
        if rule.get("holdout_by", "length") == "length_over_radius":
            return lr > rule["holdout_lr_min"]
        return length >= rule["holdout_length_min"]
    rng = row.get("psi." + cfg["protrusion"].get("range_source", "psi_range"))
    return bool(pd.notna(rng) and rng > cfg["protrusion"]["range_threshold"])


def _bucket(seed: int, group: str) -> float:
    h = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Split label per row ('' for rejected shapes). ``df`` needs id, ok, dup_group, source and the
    psi / load columns used by ``qualifies_protrusion_test``."""
    out = pd.Series("", index=df.index, dtype=object)
    ok_mask = df["ok"].astype(bool)
    if not ok_mask.any():
        return out
    groups_all = df["dup_group"].fillna(df["id"]).astype(str)
    # qualification counts every member that reached grouping, including members rejected later
    # (a qualifying shape that fails at grid must still hold out its accepted group-mates)
    grouped = df["dup_group"].notna() | ok_mask
    qual = df[grouped].apply(lambda r: qualifies_protrusion_test(r, cfg), axis=1)
    held = set(groups_all[grouped][qual.astype(bool)])
    groups = groups_all[ok_mask]
    sc = cfg["split"]
    edges = [sc["train"], sc["train"] + sc["val"]]
    inrange = set(groups_all[ok_mask & df.apply(lambda r: _is_procedural_protrusion(r), axis=1)])
    for idx, g in groups.items():
        if g in held:
            out[idx] = "protrusion_test"
            continue
        u = _bucket(cfg["seed"], g)
        if g in inrange:
            v = _bucket(cfg["seed"] + 1, g)
            if v < sc.get("protrusion_inrange_fraction", 0.0):
                out[idx] = "protrusion_test_inrange"
                continue
        out[idx] = "train" if u < edges[0] else ("val" if u < edges[1] else "test")
    for source, need in (sc.get("min_test_per_source") or {}).items():
        _top_up_test(out, df, groups, source, int(need), cfg["seed"])
    return out


def _is_procedural_protrusion(row: pd.Series) -> bool:
    return row["source"] == "procedural" and row.get("load.family") == "protrusion"


def _top_up_test(out: pd.Series, df: pd.DataFrame, groups: pd.Series, source: str, need: int, seed: int) -> None:
    """Move whole train groups containing ``source`` shapes to test, lowest hash first, until test holds at
    least ``need`` shapes of that source (or no train group of that source is left)."""
    src = df["source"] == source
    have = int(((out == "test") & src).sum())
    if have >= need:
        return
    train_groups = sorted({groups[i] for i in groups.index if out[i] == "train" and src[i]},
                          key=lambda g: _bucket(seed + 2, g))
    for g in train_groups:
        members = groups.index[groups == g]
        out[members] = "test"
        have += int(src[members].sum())
        if have >= need:
            break
