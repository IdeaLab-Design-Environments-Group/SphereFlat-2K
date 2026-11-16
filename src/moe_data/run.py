"""Run the dataset pipeline: ``uv run python -m moe_data.run --config config.yaml [--pilot]``.

Phase A (per shape): load -> topology -> suitability -> normalize, then shape descriptors.
Dedup (whole dataset): exact hash / near descriptor / same thing_id groups.
Phase B (per kept shape): remesh -> spheremap -> validate_sphere -> curvature -> ... (PER_SHAPE_B).
Every stage is cached by content key (see runner.py), so a rerun resumes where it stopped.
"""
from __future__ import annotations

# torch must be imported before pymeshlab: pymeshlab bundles its own libomp, and importing it first makes
# torch ops segfault (exit 139, reproduced on this machine). Workers import this module first.
import torch  # noqa: F401

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from moe_data import dedup, geometry, procedural, splits, thingi
from moe_data.config import load_config
from moe_data.dataset import write_processed
from moe_data.runner import StageSpec, load_stage_data, root_key, run_chain

GEOM = ("geometry.py",)

PHASE_A_THINGI = [
    StageSpec("moe_data.stages.load", (), ("thingi.py",), store_data=False),
    StageSpec("moe_data.stages.topology", (), ("topology.py",)),
    StageSpec("moe_data.stages.suitability", ("suitability",)),
    StageSpec("moe_data.stages.normalize", ()),
]
PHASE_A_PROC = [StageSpec("moe_data.stages.generate", ("seed", "procedural"), ("procedural.py",), store_data=False)] + PHASE_A_THINGI[1:]

PER_SHAPE_B = [
    StageSpec("moe_data.stages.corners", ("corners",), GEOM),
    StageSpec("moe_data.stages.remesh", ("remesh",), ("topology.py",)),
    StageSpec("moe_data.stages.spheremap", ("spheremap",), ("io_ply.py",), external=("spheremap_bin",)),
    StageSpec("moe_data.stages.validate_sphere", ("validate",), GEOM),
    StageSpec("moe_data.stages.curvature", ("curvature",), GEOM),
    StageSpec("moe_data.stages.psi", ("protrusion", "corners"), GEOM),
    StageSpec("moe_data.stages.grid", ("grid",), ("grid.py", "geometry.py")),
    StageSpec("moe_data.stages.visibility", ("visibility", "seed"), GEOM),
]


class StageHealthError(RuntimeError):
    """A dependency or stage failed for a large share of shapes: stop instead of writing a broken dataset."""


def preflight(cfg: dict) -> None:
    """Fail before any work if a required external tool is unusable."""
    binary = Path(cfg["paths"]["spheremap_bin"])
    if not (binary.is_file() and os.access(binary, os.X_OK)):
        raise StageHealthError(f"SphereMap binary not found or not executable: {binary} (run scripts/build_mreg.sh)")
    import pymeshlab
    if len(pymeshlab.filter_list()) == 0:
        raise StageHealthError("pymeshlab loaded no filters (its plugin files are unreadable or flagged hidden)")


def check_stage_health(rows: list[dict], max_crash_share: float) -> None:
    """Raise if any stage crashed (rather than rejected) for more than ``max_crash_share`` of the shapes."""
    crashes: dict[str, int] = {}
    for r in rows:
        if not r.get("ok") and str(r.get("reject_reason", "")).startswith("crash:"):
            crashes[r["reject_stage"]] = crashes.get(r["reject_stage"], 0) + 1
    bad = {st: n for st, n in crashes.items() if n > max_crash_share * len(rows)}
    if bad:
        detail = "; ".join(f"{st}: {n} of {len(rows)} shapes crashed" for st, n in sorted(bad.items()))
        example = next(r["reject_reason"] for r in rows if r.get("reject_stage") in bad
                       and str(r.get("reject_reason", "")).startswith("crash:"))
        raise StageHealthError(f"{detail}. Example: {example}")


def classify_sphere_rejections(df: pd.DataFrame, collapse_edge: float) -> pd.DataFrame:
    """Flag flipped-triangle rejections (kept in the manifest) as geometric folds or as collapse.

    When the shortest sphere edge is below ``collapse_edge`` the region is compressed to the float64 limit and
    the negative orientations are rounding noise; such shapes are rejected as unrepresentable, not folded.
    """
    flipped = (df["reject_stage"] == "validate_sphere") & \
        df["reject_reason"].fillna("").str.contains("flipped_triangles")
    edge = pd.to_numeric(df.get("validate_sphere.min_sphere_edge"), errors="coerce")
    collapsed = flipped & (edge < collapse_edge)
    return df.assign(sphere_folded=flipped & ~collapsed, sphere_collapsed=collapsed)


def _candidates(cfg: dict, pilot: bool) -> list[tuple[str, dict, list[StageSpec]]]:
    out = [(c["shape_id"], {**c, "source": "thingi10k"}, PHASE_A_THINGI)
           for c in thingi.list_candidates(cfg, pilot)]
    for sid, fam, idx in procedural.iter_specs(cfg, pilot):
        out.append((sid, {"shape_id": sid, "source": "procedural", "family": fam, "index": idx,
                          "licence": "generated", "redistributable": True}, PHASE_A_PROC))
    return out


def _phase_a(sid: str, cand: dict, specs: list[StageSpec], cfg: dict, work: str) -> dict:
    row = run_chain(sid, {"candidate": cand}, specs, cfg, Path(work), upstream_key=root_key(cand))
    row.update(source=cand["source"], licence=cand.get("licence"),
               redistributable=cand.get("redistributable"), thing_id=cand.get("thing_id"),
               original_file=cand.get("file_path", ""), self_intersecting_meta=cand.get("self_intersecting"))
    if row["ok"]:
        d = load_stage_data(Path(work), "normalize", sid)
        rng = np.random.default_rng([cfg["seed"], 7])          # same samples for every shape
        row["_descriptor"] = dedup.descriptor(d["V"], d["F"], rng).tolist()
        row["_hash"] = dedup.exact_hash(d["V"], d["F"], cfg["dedup"]["exact_round_decimals"])
        row["_n_faces"] = int(d["F"].shape[0])
    return row


def _phase_b(sid: str, upstream_key: str, cfg: dict, work: str) -> dict:
    seed = load_stage_data(Path(work), "normalize", sid)
    return run_chain(sid, seed, PER_SHAPE_B, cfg, Path(work), upstream_key=upstream_key)


def _assemble(sid: str, work: str, processed: str, psi_source: str) -> str:
    """Merge the per-stage outputs of one accepted shape into data/processed/<id>.npz."""
    d: dict = {}
    for spec in PER_SHAPE_B:
        d.update(load_stage_data(Path(work), spec.name, sid))
    d["V"] = d.pop("V_sub")                       # the float32-rounded mesh SphereMap actually mapped
    d["area_sphere"] = geometry.vertex_areas_barycentric(d["P"], d["F"])
    keep = ("V", "F", "P", "area", "area_sphere", "psi", "psi_lsq", "K_int", "K_pt", "H",
            "corner_dist", "vis", "vis_kind", "vis_dirs", "grid", "grid_channels")
    path = Path(processed) / f"{sid}.npz"
    out = {k: d[k] for k in keep}
    out["grid_psi_source"] = np.array(psi_source)          # which psi estimator built the stored grid
    write_processed(path, out)
    return str(path)


def _dedup(passed: list[dict], cfg: dict) -> dict:
    """Exact-hash duplicates across all sources; near-duplicates (descriptor distance) only among Thingi10K.

    Procedural shapes are drawn from independent parameters, so a descriptor match between two of them is a
    false positive (a 2- and a 3-protrusion shape can be 0.014 apart).
    """
    if not passed:
        return {}
    recs = {}
    for source, thr in (("thingi10k", cfg["dedup"]["near_threshold"]), ("procedural", -1.0)):
        rows = [r for r in passed if r["source"] == source]
        if rows:
            recs.update(dedup.dedup_records(
                [r["id"] for r in rows], np.array([r["_descriptor"] for r in rows]), [r["_hash"] for r in rows],
                [r["thing_id"] if source == "thingi10k" else None for r in rows], thr,
                [r["_n_faces"] for r in rows]))
    hashes = {}                                     # exact duplicates across the two sources
    for r in passed:
        hashes.setdefault(r["_hash"], []).append(r["id"])
    merge = {}                                      # group id -> merged group id (union-find)

    def find(g):
        while merge.get(g, g) != g:
            g = merge[g]
        return g
    for ids in hashes.values():
        if len({i.split("_")[0] for i in ids}) > 1:
            keep = sorted(ids)[0]
            root = find(recs[keep]["group_id"])
            for i in ids:
                merge[find(recs[i]["group_id"])] = root
                if i != keep:
                    recs[i].update(keep=False, reject_stage="dedup", reject_reason=f"duplicate_of: {keep}")
    for rec in recs.values():
        rec["group_id"] = find(rec["group_id"])
    return recs


def _fallback_queues(recs: dict, passed: list[dict]) -> dict[str, list[str]]:
    """Representative id -> the other members of its duplicate component, in the order they should be
    tried if the representative fails phase B (most faces first, then id), so one failed representative
    does not silently remove its whole component."""
    nf = {r["id"]: r["_n_faces"] for r in passed}
    queues: dict[str, list[str]] = {}
    for i, rec in recs.items():
        reason = rec.get("reject_reason", "")
        if not rec["keep"] and reason.startswith("duplicate_of: "):
            queues.setdefault(reason.removeprefix("duplicate_of: "), []).append(i)
    return {rep: sorted(m, key=lambda i: (-nf[i], i)) for rep, m in queues.items()}


def _pool_map(fn, jobs: list[tuple], cfg: dict, label: str) -> list[dict]:
    rows, t0 = [], time.time()
    ctx = get_context("spawn")
    # No max_tasks_per_child: on Python 3.12 the executor deadlocks once every worker has hit the limit
    # (the Thingi10K survey stopped at exactly 12 x 25 = 300 shapes; tests/test_pool.py).
    with ProcessPoolExecutor(max_workers=cfg["run"]["max_workers"], mp_context=ctx) as ex:
        futs = {ex.submit(fn, *j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:  # worker died: record, never drop silently
                rows.append({"id": futs[f], "ok": False, "reject_stage": "worker",
                             "reject_reason": f"crash: {type(e).__name__}: {e}"})
            if i % 25 == 0 or i == len(jobs):
                print(f"[{label}] {i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tag = "pilot" if args.pilot else "full"
    work = Path(cfg["paths"]["work"]) / tag
    work.mkdir(parents=True, exist_ok=True)
    cfg = {**cfg, "_tag": tag}

    preflight(cfg)
    max_crash = cfg["run"]["max_crash_share"]
    cands = _candidates(cfg, args.pilot)
    os.environ["HF_HUB_OFFLINE"] = "1"                        # cache exists from here on
    rows_a = _pool_map(_phase_a, [(s, c, sp, cfg, str(work)) for s, c, sp in cands], cfg, "A")

    check_stage_health(rows_a, max_crash)
    passed = [r for r in rows_a if r["ok"]]
    recs = _dedup(passed, cfg)
    for r in passed:
        rec = recs[r["id"]]
        r.update(dup_group=rec["group_id"], dup_group_reason=rec["dup_group_reason"])
        if not rec["keep"]:
            r.update(ok=False, reject_stage="dedup", reject_reason=rec["reject_reason"])

    keep = [r for r in passed if r["ok"]]
    rows_b = {r["id"]: r for r in _pool_map(_phase_b, [(r["id"], r["key"], cfg, str(work)) for r in keep],
                                            cfg, "B")}
    check_stage_health(list(rows_b.values()), max_crash)
    # representative fallback: while a component's current representative failed phase B, try the next
    # member; the failed representative keeps its own phase-B rejection, the others point at the new one
    by_id = {r["id"]: r for r in passed}
    queues = _fallback_queues(recs, passed)
    current = {rep: rep for rep in queues}
    while True:
        todo = [(rep, queues[rep].pop(0)) for rep in queues
                if not rows_b[current[rep]]["ok"] and queues[rep]]
        if not todo:
            break
        new = _pool_map(_phase_b, [(m, by_id[m]["key"], cfg, str(work)) for _, m in todo], cfg, "B-fallback")
        rows_b.update({r["id"]: r for r in new})
        for rep, m in todo:
            current[rep] = m
            by_id[m].update(ok=True, reject_stage="", reject_reason="", dedup_fallback_for=rep)
    for rep, cur in current.items():
        if cur != rep:
            for m in queues[rep] + [rep]:
                r = by_id[m]
                if m != rep and r.get("reject_stage") == "dedup":
                    r["reject_reason"] = f"duplicate_of: {cur}"
    manifest = []
    for r in rows_a:
        r = {k: v for k, v in r.items() if not k.startswith("_")}
        b = rows_b.get(r["id"])
        if b is not None:
            r.update({k: v for k, v in b.items() if k not in ("id",)})
        manifest.append(r)
    df = pd.DataFrame(manifest)
    df = classify_sphere_rejections(df, cfg["run"]["collapse_edge"])
    df["split"] = splits.assign_splits(df, cfg)

    processed = Path(cfg["paths"]["processed"]) / tag
    processed.mkdir(parents=True, exist_ok=True)
    df["processed_file"] = ""
    for i in df.index[df["ok"].astype(bool)]:
        df.at[i, "processed_file"] = _assemble(df.at[i, "id"], str(work), str(processed), cfg["grid"]["psi_source"])

    out = Path(cfg["paths"]["reports"]) / tag
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.parquet" if args.pilot else Path(cfg["paths"]["manifest"])
    df.to_parquet(manifest_path, index=False)
    print(json.dumps({"manifest": str(manifest_path), "candidates": len(df), "ok": int(df["ok"].sum()),
                      "rejected_by_stage": df.loc[~df["ok"], "reject_stage"].value_counts().to_dict(),
                      "splits": df.loc[df["ok"], "split"].value_counts().to_dict()}, indent=1))


if __name__ == "__main__":
    main()
