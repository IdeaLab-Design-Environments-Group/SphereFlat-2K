"""Pilot / full-run report.

``uv run python -m moe_data.report --config config.yaml --manifest <manifest.parquet> --processed <dir>
--out <dir> [--survey data/reports/survey] [--work <dir>] [--survey-meshes <dir>] [--final auto|yes|no]
[--stretch <STRETCH.md>]``

For the full run (manifest at paths.manifest or processed dir ``.../full``) the report is titled the final
report and leads with the dataset size, folded-map rejections, ranked rejection reasons and items to inspect
by hand; the decisions section then lists only the PROPOSED thresholds that remain.

Writes ``<out>/REPORT.md`` and PNG figures in ``<out>/fig/``. Every number is computed from the manifest,
the processed files, the work cache, the optional Thingi10K phase-A survey (``phase_a.parquet``,
``descriptors.npz``) and config.yaml. A missing input or column is reported as "not available".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from moe_data.config import load_config  # noqa: E402

NA = "not available"
STAGE_ORDER = ("load", "topology", "suitability", "normalize", "dedup", "remesh", "spheremap",
               "validate_sphere", "curvature", "psi", "grid", "visibility")
PHASE_A = ("load", "topology", "suitability", "normalize")
# Thingi10K topology rejects ~75% by construction (metadata: ~2.5k of 10k files are closed genus-0 manifolds).
EXPECTED_HIGH_REJECTION = {("thingi10k", "topology")}
FLAG_SHARE = 0.5
DEDUP_TABLE_THRESHOLDS = (0.002, 0.005, 0.01, 0.02)
DEDUP_PAIR_BINS = ((0.0, 0.003), (0.003, 0.006), (0.006, 0.01), (0.01, 0.02))
PAIRS_PER_BIN = 3
PAIR_MAX_FACES = 20000
N_SHAPES_PER_SIDE = 5
GROUP_COLORS = {"thingi10k": "#4C78A8", "proc_blob": "#F58518", "proc_protrusion": "#54A24B"}
GROUPS = tuple(GROUP_COLORS)
SPLIT_LABELS = ("train", "val", "test", "protrusion_test", "protrusion_test_inrange")
THRESHOLD_PAIRS_PER_SIDE = 6
CURV_SAMPLE_PER_GROUP = 200
N_INSPECT = 5
BIG_DUP_GROUP = 5
DPI = 110


# ----------------------------------------------------------------------------------------------- helpers

def _f(x, digits: int = 3) -> str:
    if x is None:
        return NA
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return NA
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
        return f"{x:.{digits}g}"
    return f"{x:.{digits}g}" if abs(x) < 1000 else f"{x:,.0f}"


def _pct(x) -> str:
    return NA if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
    return out + [""]


def _num(df: pd.DataFrame | None, col: str) -> pd.Series:
    """Float column, or an empty series when the frame or column is missing."""
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _flag_true(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].map(lambda v: v is True or v is np.True_ or (isinstance(v, str) and v == "True"))


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return f"fig/{path.name}"


def group_labels(df: pd.DataFrame) -> pd.Series:
    """thingi10k / proc_<family>, from source + load.family, falling back to the id prefix."""
    ids = df["id"].astype(str)
    src = df["source"].astype(str) if "source" in df.columns else pd.Series("", index=df.index)
    fam = df["load.family"] if "load.family" in df.columns else pd.Series(np.nan, index=df.index)
    out = []
    for i, s, f in zip(ids, src, fam):
        if s == "thingi10k" or i.startswith("thingi_"):
            out.append("thingi10k")
        elif isinstance(f, str) and f:
            out.append(f"proc_{f}")
        elif i.startswith("proc_prot"):
            out.append("proc_protrusion")
        elif i.startswith("proc_blob"):
            out.append("proc_blob")
        else:
            out.append(s or "unknown")
    return pd.Series(out, index=df.index)


def group_order(labels) -> list[str]:
    """The known groups first (fixed order), then any others sorted."""
    present = set(labels)
    return [g for g in GROUPS if g in present] + sorted(present - set(GROUPS))


def quantile_cells(x: pd.Series) -> list:
    """[n, min, p5, median, p95, max] of the finite values (NA cells when empty)."""
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if not len(x):
        return [0] + [NA] * 5
    return [len(x)] + [_f(v) for v in (x.min(), x.quantile(0.05), x.median(), x.quantile(0.95), x.max())]


def folded_mask(df: pd.DataFrame) -> pd.Series:
    """The manifest's sphere_folded flag; older manifests without it: validate_sphere rejections whose
    reason contains flipped_triangles (the same rule run.py applies)."""
    if "sphere_folded" in df.columns:
        return df["sphere_folded"].fillna(False).astype(bool)
    rs = df["reject_stage"].fillna("") if "reject_stage" in df.columns else pd.Series("", index=df.index)
    rr = df["reject_reason"].fillna("").astype(str) if "reject_reason" in df.columns \
        else pd.Series("", index=df.index)
    return (rs == "validate_sphere") & rr.str.contains("flipped_triangles")


def threshold_pairs(pairs: list[tuple[int, int, float]], thr: float, k: int) -> tuple[list, list]:
    """From (i, j, distance) pairs: the k highest distances below ``thr`` (closest to it first) and the k
    lowest distances at or above it."""
    srt = sorted(pairs, key=lambda p: p[2])
    below = [p for p in srt if p[2] < thr][-k:][::-1] if k > 0 else []
    above = [p for p in srt if p[2] >= thr][:k]
    return below, above


def is_final_run(cfg: dict, manifest_path: str | Path, processed: Path) -> bool:
    """True for the full run: manifest at paths.manifest, or processed files under <paths.processed>/full."""
    paths = cfg.get("paths", {})
    m = paths.get("manifest")
    if m and Path(manifest_path).resolve() == Path(m).resolve():
        return True
    return processed.name == "full"


def dir_bytes(path: Path | None) -> tuple[int, int]:
    """(total bytes, file count) under a directory, recursively; (0, 0) when missing."""
    if path is None or not path.is_dir():
        return 0, 0
    total = n = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
                n += 1
            except OSError:                 # file removed while scanning (run in progress)
                pass
    return total, n


def reject_stages(df: pd.DataFrame) -> pd.Series:
    ok = df["ok"].astype(bool) if "ok" in df.columns else pd.Series(False, index=df.index)
    rs = df["reject_stage"].fillna("").astype(str) if "reject_stage" in df.columns else pd.Series("", index=df.index)
    return rs.where(~ok, "").where(ok | (rs != ""), "unknown")


def stage_list(*frames: pd.DataFrame) -> list[str]:
    extra = sorted({s for df in frames if df is not None for s in reject_stages(df) if s and s not in STAGE_ORDER})
    return list(STAGE_ORDER) + extra


def stage_flow(df: pd.DataFrame, stages: list[str]) -> list[dict]:
    """Entering / rejected / surviving count per stage, in pipeline order."""
    rs = reject_stages(df)
    n, out = len(df), []
    for s in stages:
        rej = int((rs == s).sum())
        out.append({"stage": s, "in": n, "rejected": rej, "out": n - rej})
        n -= rej
    return out


def reason_code(reason: str) -> str:
    """'a: 12 x; b: 3' -> 'a + b' (the machine-readable codes, numbers stripped)."""
    parts = [p.split(":")[0].strip() for p in str(reason).split("; ") if p.strip()]
    return " + ".join(parts) if parts else "(empty)"


def parse_proposed(config_path: str | Path) -> list[tuple[str, str]]:
    """(section, key) pairs marked PROPOSED in config.yaml; key '*' means the whole section."""
    out, section = [], None
    for line in Path(config_path).read_text().splitlines():
        m = re.match(r"^([A-Za-z_]\w*):", line)
        if m:
            section = m.group(1)
        if "PROPOSED" not in line.split("#", 1)[-1] or "#" not in line:
            continue
        if m:
            out.append((section, "*"))
            continue
        k = re.match(r"^\s+([A-Za-z_]\w*):", line)
        if k and section:
            out.append((section, k.group(1)))
    return out


# --------------------------------------------------------------------------------------------- inputs

def load_survey(survey_dir: str | Path | None) -> tuple[pd.DataFrame | None, dict | None]:
    if not survey_dir:
        return None, None
    d = Path(survey_dir)
    df = pd.read_parquet(d / "phase_a.parquet") if (d / "phase_a.parquet").exists() else None
    desc = None
    if (d / "descriptors.npz").exists():
        with np.load(d / "descriptors.npz", allow_pickle=False) as z:
            desc = {k: z[k] for k in z.files}
    return df, desc


def scan_work(work: Path | None) -> dict[str, int]:
    """Bytes of work-cache files per shape id (all stages)."""
    sizes: dict[str, int] = {}
    if work is None or not work.is_dir():
        return sizes
    for stage_dir in work.iterdir():
        if not stage_dir.is_dir():
            continue
        with os.scandir(stage_dir) as it:
            for e in it:
                if e.is_file():
                    sid = e.name.split(".")[0]
                    sizes[sid] = sizes.get(sid, 0) + e.stat().st_size
    return sizes


def _read_npz(path: Path, keys: tuple[str, ...]) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys if k in z.files}


# ---------------------------------------------------------------------------------------- histograms

def _hist_panel(ax, groups: dict[str, np.ndarray], thr: float | None, alts: list[float], side: str,
                log: bool, xlabel: str, title: str, verb: str = "rejected",
                floor: float | None = None) -> list[dict] | None:
    """Stacked histogram by group, the current threshold (solid) and alternatives (dashed) with the
    share of all values on the rejecting side ('below' or 'above'). Returns the share rows."""
    groups = {g: np.asarray(v, float)[np.isfinite(np.asarray(v, float))] for g, v in groups.items()}
    groups = {g: v for g, v in groups.items() if v.size}
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    if not groups:
        ax.text(0.5, 0.5, NA, ha="center", va="center", transform=ax.transAxes)
        return None
    allv = np.concatenate(list(groups.values()))
    n_floor = 0
    if floor is not None:              # draw values below the floor (incl. 0) in the first bin
        n_floor = int(sum(np.sum(v < floor) for v in groups.values()))
        shown = {g: np.maximum(v, floor) for g, v in groups.items()}
    else:
        shown = {g: (v[v > 0] if log else v) for g, v in groups.items()}
    n_hidden = int(sum(len(groups[g]) - len(shown[g]) for g in groups))
    pos = np.concatenate(list(shown.values())) if shown else np.array([])
    if pos.size:
        lo, hi = pos.min(), pos.max()
        extra = [t for t in [thr, *alts] if t is not None and (t > 0 or not log)]
        lo, hi = min([lo, *extra]), max([hi, *extra])
        if hi <= lo:
            hi = lo + (abs(lo) if lo else 1.0)
        bins = np.geomspace(lo, hi, 40) if log else np.linspace(lo, hi, 40)
        names = [g for g in shown if shown[g].size]
        ax.hist([shown[g] for g in names], bins=bins, stacked=True, label=names,
                color=[GROUP_COLORS.get(g, "#999999") for g in names])
    if log:
        ax.set_xscale("log")
    rows = []
    for t, cur in ([(thr, True)] if thr is not None else []) + [(a, False) for a in alts]:
        share = float(np.mean(allv < t) if side == "below" else np.mean(allv > t))
        rows.append({"value": t, "current": cur, "share": share, "n": int(allv.size),
                     "per_group": {g: float(np.mean(v < t) if side == "below" else np.mean(v > t))
                                   for g, v in groups.items()}})
        if not (log and t <= 0):
            ax.axvline(t, color="black" if cur else "0.45", ls="-" if cur else "--", lw=1.4 if cur else 1.0,
                       label=f"{'current ' if cur else ''}{t:g}: {100 * share:.1f}% {verb}")
    if n_floor:
        ax.text(0.98, 0.95, f"{n_floor} values < {floor:g} drawn at {floor:g}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7)
    if n_hidden:
        ax.text(0.98, 0.95, f"{n_hidden} values <= 0 not shown", transform=ax.transAxes, ha="right",
                va="top", fontsize=7)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6.5, loc="best")
    ax.set_ylabel("shapes", fontsize=8)
    return rows


def _share_rows(name: str, rows: list[dict] | None, verb: str) -> list[list]:
    if not rows:
        return [[name, NA, NA, NA, NA]]
    out = []
    for r in rows:
        pg = ", ".join(f"{g} {_pct(s)}" for g, s in r["per_group"].items())
        out.append([name, f"{r['value']:g}" + (" (current)" if r["current"] else ""), r["n"],
                    f"{_pct(r['share'])} {verb}", pg])
    return out


def _by_group(df: pd.DataFrame, col: str) -> dict[str, np.ndarray]:
    if col not in df.columns:
        return {}
    g = group_labels(df)
    v = _num(df, col)
    return {k: v[g == k].dropna().to_numpy() for k in sorted(g.unique()) if v[g == k].notna().any()}


# ------------------------------------------------------------------------------------------ the report

class Report:
    def __init__(self, cfg: dict, config_path: str, manifest: pd.DataFrame, processed: Path, out: Path,
                 survey_df: pd.DataFrame | None, survey_desc: dict | None, work: Path | None,
                 survey_meshes: Path | None, final: bool = False, stretch: Path | None = None):
        self.cfg, self.config_path, self.df = cfg, config_path, manifest
        self.processed, self.out, self.fig = processed, out, out / "fig"
        self.survey_df, self.desc, self.work, self.survey_meshes = survey_df, survey_desc, work, survey_meshes
        self.final, self.stretch = final, stretch
        self.df["_group"] = group_labels(self.df)
        self.groups = group_order(self.df["_group"])
        self.ok = self.df["ok"].astype(bool) if "ok" in self.df.columns else pd.Series(False, index=self.df.index)
        self.stages = stage_list(self.df, survey_df)
        self.decisions: list[str] = []
        self.threshold_summary: dict[tuple[str, str], str] = {}
        self.flags: list[str] = []
        self.projection: dict = {}
        self.nn: dict | None = None

    # ---- 1. counts
    def counts(self) -> list[str]:
        L = ["## 1. Counts", ""]
        df = self.df
        L.append("Candidates per source / group (this manifest):")
        L.append("")
        vc = df.groupby(["source" if "source" in df.columns else "_group", "_group"]).size()
        L += _table(["source", "group", "candidates", "accepted"],
                    [[s, g, int(n), int(self.ok[(df["_group"] == g)].sum())] for (s, g), n in vc.items()]
                    + [["all", "", len(df), int(self.ok.sum())]])
        L.append("Survivors after each stage per group and overall, pipeline order (normalize sits between "
                 "suitability and dedup in run.py). Flag: a stage rejecting > 50% of its input"
                 + " (Thingi10K topology exempt).")
        L.append("")
        rows = []
        for grp in self.groups + ["all"]:
            sub = df if grp == "all" else df[df["_group"] == grp]
            for r in stage_flow(sub, self.stages):
                if r["in"] == 0 and r["rejected"] == 0:
                    continue
                share = r["rejected"] / r["in"] if r["in"] else float("nan")
                flag = ""
                if (grp != "all" and r["in"] and share > FLAG_SHARE
                        and (grp, r["stage"]) not in EXPECTED_HIGH_REJECTION):
                    flag = "check: bug or threshold?"
                    self.flags.append(f"{grp} / {r['stage']}: {r['rejected']} of {r['in']} rejected "
                                      f"({_pct(share)})")
                rows.append([grp, r["stage"], r["in"], r["rejected"], r["out"], _pct(share), flag])
        L += _table(["group", "stage", "in", "rejected", "survivors", "rejected share", "flag"], rows)
        if len(df) and self.stages:
            final = stage_flow(df, self.stages)[-1]["out"]
            if final != int(self.ok.sum()):
                L.append(f"Inconsistency: {final} survive all stages but {int(self.ok.sum())} rows have ok=True.")
                L.append("")
        L += self.reasons_ranked()

        if self.survey_df is not None:
            L.append(f"Full Thingi10K phase-A survey ({len(self.survey_df):,} files):")
            L.append("")
            rows = []
            for r in stage_flow(self.survey_df, [s for s in self.stages if s in PHASE_A]):
                share = r["rejected"] / r["in"] if r["in"] else float("nan")
                rows.append([r["stage"], r["in"], r["rejected"], r["out"], _pct(share)])
            L += _table(["stage", "in", "rejected", "survivors", "rejected share"], rows)
            sv = self.survey_df[~self.survey_df["ok"].astype(bool)]
            codes = sv["reject_reason"].map(reason_code).value_counts()
            L += _table(["survey reason code", "count"], [[c, int(n)] for c, n in codes.head(15).items()])
        return L

    def reasons_ranked(self) -> list[str]:
        """Rejection reasons ranked by frequency: overall, then one table per group."""
        df = self.df
        rej = df[~self.ok].copy()
        L = ["Rejection reasons ranked by frequency (code = text before ':'; multiple failures joined with +), "
             "overall:", ""]
        if not len(rej) or "reject_reason" not in rej.columns:
            return L + ["No rejections (or no reject_reason column).", ""]
        rej["_stage"] = reject_stages(rej)
        rej["_code"] = rej["reject_reason"].map(reason_code)

        def ranked(frame: pd.DataFrame, with_groups: bool) -> list[list]:
            rows = []
            for (s, c), sub in sorted(frame.groupby(["_stage", "_code"]), key=lambda kv: (-len(kv[1]), kv[0])):
                row = [s, c, len(sub), _pct(len(sub) / len(frame))]
                if with_groups:
                    row.append(", ".join(f"{k} {v}" for k, v in sub["_group"].value_counts().items()))
                rows.append(row + [f"`{sub['id'].iloc[0]}`: {sub['reject_reason'].iloc[0]}"])
            return rows

        L += _table(["stage", "reason code", "count", "share of rejections", "by group", "example"],
                    ranked(rej, True))
        for g in self.groups:
            sub = rej[rej["_group"] == g]
            n = int((df["_group"] == g).sum())
            L += [f"{g}: {len(sub):,} of {n:,} candidates rejected.", ""]
            if len(sub):
                L += _table(["stage", "reason code", "count", "share of group rejections", "example"],
                            ranked(sub, False))
        return L

    def _grouped(self, col: str, frame: pd.DataFrame | None = None, absval: bool = False) -> dict[str, pd.Series]:
        """Values of ``col`` per group plus 'all' (rows where it was recorded)."""
        frame = self.df if frame is None else frame
        x = _num(frame, col)
        if absval:
            x = x.abs()
        gl = frame["_group"] if "_group" in frame.columns else group_labels(frame)
        out = {g: x[gl == g].dropna() for g in group_order(gl)} if len(x) else {}
        out["all"] = x.dropna()
        return out

    def _dist_table(self, specs: list[tuple[str, dict[str, pd.Series]]]) -> list[str]:
        """Quantile table: one row per (quantity, group) with n, min, p5, median, p95, max."""
        rows = []
        for label, per in specs:
            if not per or not len(per.get("all", [])):
                rows.append([label, "all", 0] + [NA] * 5)
                continue
            for g, x in per.items():
                if len(x) or g == "all":
                    rows.append([label, g] + quantile_cells(x))
        return _table(["quantity", "group", "n", "min", "p5", "median", "p95", "max"], rows)

    # ---- 2. thresholds
    def _suitability_frame(self) -> tuple[pd.DataFrame, str]:
        proc = self.df[self.df["_group"] != "thingi10k"]
        if self.survey_df is not None and "suitability.n_faces" in self.survey_df.columns:
            s = self.survey_df.copy()
            s["source"] = "thingi10k"
            return pd.concat([s, proc], ignore_index=True), "Thingi10K from the full survey, procedural from this manifest"
        return self.df, "this manifest only (no survey)"

    def thresholds(self) -> list[str]:
        cfg = self.cfg
        L = ["## 2. Threshold distributions", "",
             "Shares are over shapes that reached the stage (measure recorded). Suitability checks run in "
             "sequence (n_faces, compactness, pca); each share here is for that check alone.", ""]
        rows: list[list] = []

        # suitability
        frame, src_note = self._suitability_frame()
        frame = frame.copy()
        frame["_group"] = group_labels(frame)
        su = cfg.get("suitability", {})
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        specs = [("suitability.n_faces", "min_faces", "n_faces", True, (0.5, 2, 4)),
                 ("suitability.compactness", "min_compactness", "compactness V/A^1.5", False, (0.5, 2, 4)),
                 ("suitability.pca_ratio_min", "min_pca_ratio", "pca_ratio_min", False, (0.5, 2, 5))]
        for ax, (col, key, label, log, mult) in zip(axs, specs):
            thr = su.get(key)
            alts = [thr * m for m in mult] if thr is not None else []
            r = _hist_panel(ax, _by_group(frame, col), thr, alts, "below", log, label + (" (log)" if log else ""),
                            f"suitability.{key}")
            rows += _share_rows(f"suitability.{key}", r, "rejected")
            self._summ("suitability", key, r, "rejected")
        fig.suptitle(f"Suitability measures ({src_note})", fontsize=10)
        f_suit = _save(fig, self.fig / "suitability.png")

        # remesh hausdorff
        fig, ax = plt.subplots(figsize=(6, 4))
        tol = cfg.get("remesh", {}).get("hausdorff_tol")
        r = _hist_panel(ax, _by_group(self.df, "remesh.hausdorff_sym"), tol,
                        [tol * m for m in (0.5, 2, 3)] if tol else [], "above", True,
                        "symmetric Hausdorff / bbox diagonal (log)", "remesh.hausdorff_tol")
        rows += _share_rows("remesh.hausdorff_tol", r, "rejected")
        self._summ("remesh", "hausdorff_tol", r, "rejected")
        f_haus = _save(fig, self.fig / "remesh_hausdorff.png")

        # psi range / area share low
        pc = cfg.get("protrusion", {})
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        rt = pc.get("range_threshold")
        rsrc = pc.get("range_source", "psi_range")
        r = _hist_panel(axs[0], _by_group(self.df, f"psi.{rsrc}"), rt,
                        [rt * m for m in (0.5, 1.5, 2)] if rt else [], "above", False,
                        f"{rsrc}: psi max - min (relative to area-weighted mean)", "protrusion.range_threshold",
                        verb="tagged")
        rows += _share_rows("protrusion.range_threshold", r, "tagged protrusion")
        self._summ("protrusion", "range_threshold", r, "tagged protrusion (Thingi10K -> protrusion_test)")
        _hist_panel(axs[1], _by_group(self.df, "psi.psi_area_share_low"), None, [], "above", False,
                    f"area share with psi_rel < psi_low = {pc.get('psi_low')}", "protrusion.psi_low (area share)")
        f_psi = _save(fig, self.fig / "psi_protrusion.png")
        psi_low_rows = self._psi_low_alternatives(pc.get("psi_low"))

        # validate min_sphere_edge
        fig, ax = plt.subplots(figsize=(6, 4))
        me = cfg.get("validate", {}).get("min_sphere_edge")
        r = _hist_panel(ax, _by_group(self.df, "validate_sphere.min_sphere_edge"), me,
                        [me * m for m in (0.1, 10, 100)] if me else [], "below", True,
                        "shortest sphere edge (chord, log)", "validate.min_sphere_edge")
        rows += _share_rows("validate.min_sphere_edge", r, "rejected")
        f_edge = _save(fig, self.fig / "validate_min_sphere_edge.png")

        # procedural protrusions: psi range vs max L/r
        f_lr = self._lr_scatter()

        # dedup near-neighbour distances
        dedup_lines = self._dedup_section()

        L += _table(["threshold", "value", "n", "share", "share per group"], rows)
        L += ["Distribution of each thresholded measure per group (all shapes where it was recorded; "
              "Thingi10K suitability from the survey when given):", ""]
        L += self._dist_table([
            ("suitability.n_faces", self._grouped("suitability.n_faces", frame)),
            ("suitability.compactness", self._grouped("suitability.compactness", frame)),
            ("suitability.pca_ratio_min", self._grouped("suitability.pca_ratio_min", frame)),
            ("remesh.hausdorff_sym", self._grouped("remesh.hausdorff_sym")),
            ("validate_sphere.min_sphere_edge", self._grouped("validate_sphere.min_sphere_edge")),
            (f"psi.{rsrc} (range_source)", self._grouped(f"psi.{rsrc}")),
        ])
        L += [f"![suitability]({f_suit})", "", f"![hausdorff]({f_haus})", "",
              f"![psi protrusion]({f_psi})", ""]
        L.append(f"Area share with psi_rel below alternative psi_low values (from processed files, "
                 f"psi = brief estimator, barycentric areas of V_sub):")
        L.append("")
        L += psi_low_rows
        L += [f"![min sphere edge]({f_edge})", ""]
        L += [f"![protrusion L/r]({f_lr})" if f_lr else f"Protrusion psi range vs L/r: {NA}.", ""]
        L += dedup_lines
        return L

    def _summ(self, section: str, key: str, rows: list[dict] | None, verb: str) -> None:
        if not rows:
            self.threshold_summary[(section, key)] = f"{NA} (no measured values)"
            return
        cur = [r for r in rows if r["current"]]
        alts = "; ".join(f"{r['value']:g} -> {_pct(r['share'])}" for r in rows if not r["current"])
        c = cur[0] if cur else None
        self.threshold_summary[(section, key)] = (
            (f"{_pct(c['share'])} {verb} (n = {c['n']:,})" if c else "") + f"; alternatives: {alts}")

    def _psi_low_alternatives(self, psi_low) -> list[str]:
        if psi_low is None:
            return [NA, ""]
        values = [psi_low + d for d in (1.0, 0.0, -1.0, -2.0)]
        per: dict[str, list[list[float]]] = {}
        for _, row in self.df[self.ok].iterrows():
            d = _read_npz(self.processed / f"{row['id']}.npz", ("psi", "area"))
            if not d or "psi" not in d or "area" not in d:
                continue
            a, psi = d["area"], d["psi"]
            rel = psi - np.sum(a * psi) / np.sum(a)
            per.setdefault(row["_group"], []).append([float(a[rel < v].sum() / a.sum()) for v in values])
        if not per:
            return [f"{NA} (no processed files)", ""]
        j0 = values.index(psi_low)
        self.threshold_summary[("protrusion", "psi_low")] = "area share below it: " + "; ".join(
            f"{g} median {_f(np.median(np.array(a)[:, j0]))}, max {_f(np.max(np.array(a)[:, j0]))}"
            for g, a in sorted(per.items())) + " (alternatives in section 2)"
        rows = []
        for j, v in enumerate(values):
            cells = []
            for g, arr in sorted(per.items()):
                x = np.array(arr)[:, j]
                cells.append(f"{g}: median {_f(np.median(x))}, max {_f(x.max())}, >0 in {int((x > 0).sum())}/{len(x)}")
            rows.append([f"{v:g}" + (" (current)" if v == psi_low else ""), "; ".join(cells)])
        return _table(["psi_low", "area share below (per group)"], rows)

    def _protrusion_params(self) -> pd.DataFrame:
        from moe_data.splits import procedural_severity
        if "load.params_json" not in self.df.columns:
            return pd.DataFrame()
        sub = self.df[(self.df["_group"] == "proc_protrusion") & self.df["load.params_json"].notna()].copy()
        if not len(sub):
            return pd.DataFrame()
        sev = sub["load.params_json"].map(procedural_severity)
        sub["_max_len"] = [s[0] for s in sev]
        sub["_max_lr"] = [s[1] for s in sev]
        return sub

    def _lr_scatter(self) -> str | None:
        sub = self._protrusion_params()
        if not len(sub) or "psi.psi_range" not in sub.columns:
            return None
        rng = _num(sub, "psi.psi_range")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        has = rng.notna()
        ax.scatter(sub.loc[has, "_max_lr"], rng[has], s=18, color=GROUP_COLORS["proc_protrusion"],
                   label=f"psi computed ({int(has.sum())})")
        rej = ~self.ok.loc[sub.index]
        if rej.any():
            ymin = float(rng.min()) if has.any() else 0.0
            ax.scatter(sub.loc[rej, "_max_lr"], np.full(int(rej.sum()), ymin), marker="x", color="crimson",
                       label=f"rejected ({int(rej.sum())}), drawn at y=min")
        pc, pp = self.cfg.get("protrusion", {}), self.cfg.get("split", {})
        if pc.get("range_threshold") is not None:
            ax.axhline(pc["range_threshold"], color="black", lw=1, label=f"range_threshold {pc['range_threshold']:g}")
        if pp.get("holdout_lr_min") is not None:
            ax.axvline(pp["holdout_lr_min"], color="0.4", ls="--", lw=1, label=f"holdout_lr_min {pp['holdout_lr_min']:g}")
        ax.set_xlabel("max protrusion length / thickness (L/r)", fontsize=8)
        ax.set_ylabel("psi range (relative)", fontsize=8)
        ax.set_title("Procedural protrusions: psi range vs max L/r", fontsize=9)
        ax.legend(fontsize=7)
        return _save(fig, self.fig / "protrusion_lr.png")

    # ---- dedup (survey)
    def _nn_stats(self) -> dict | None:
        if self.desc is None or not {"D", "thing", "ids"} <= set(self.desc):
            return None
        D = np.asarray(self.desc["D"], float)
        thing = np.asarray(self.desc["thing"])
        n = len(D)
        nn_d = np.full(n, np.inf)
        nn_j = np.full(n, -1)
        same: list[np.ndarray] = []
        sq = np.sum(D * D, axis=1)
        for a in range(0, n, 1024):
            b = min(n, a + 1024)
            d2 = np.maximum(sq[a:b, None] + sq[None, :] - 2 * D[a:b] @ D.T, 0.0)
            dist = np.sqrt(d2)
            st = thing[a:b, None] == thing[None, :]
            for k in range(b - a):          # same-thing pairs (upper triangle only)
                js = np.nonzero(st[k])[0]
                js = js[js > a + k]
                if js.size:
                    same.append(dist[k, js])
            dist[st] = np.inf               # also removes self
            nn_j[a:b] = np.argmin(dist, axis=1)
            nn_d[a:b] = dist[np.arange(b - a), nn_j[a:b]]
        # exact recompute of the selected distances (the Gram trick loses digits near 0)
        fin = nn_j >= 0
        nn_d[fin] = np.linalg.norm(D[fin] - D[nn_j[fin]], axis=1)
        return {"D": D, "thing": thing, "ids": np.asarray(self.desc["ids"]).astype(str), "nn_d": nn_d,
                "nn_j": nn_j, "same": np.concatenate(same) if same else np.array([])}

    def _dedup_section(self) -> list[str]:
        L = ["### Dedup: Thingi10K descriptor distances", ""]
        self.nn = nn = self._nn_stats()
        thr = self.cfg.get("dedup", {}).get("near_threshold")
        if nn is None:
            L += [f"Nearest-neighbour distances: {NA} (needs the survey descriptors.npz).", ""]
            self.threshold_summary[("dedup", "near_threshold")] = f"{NA} (no survey)"
            return L
        fin = np.isfinite(nn["nn_d"])
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        r = _hist_panel(axs[0], {"thingi10k": nn["nn_d"][fin]}, thr,
                        [thr * m for m in (0.25, 0.5, 2)] if thr else [], "below", True,
                        "nearest-neighbour distance, other thing_id (log)", "dedup.near_threshold (cross-thing NN)",
                        verb="have a neighbour below", floor=thr / 100 if thr else None)
        _hist_panel(axs[1], {"thingi10k": nn["same"]}, thr, [], "below", True,
                    "pairwise distance within the same thing_id (log)", "calibration: same-thing pairs",
                    verb="below", floor=thr / 100 if thr else None)
        axs[1].set_ylabel("pairs", fontsize=8)
        f = _save(fig, self.fig / "dedup_nn.png")
        self._summ("dedup", "near_threshold", r, "of survey shapes have a cross-thing neighbour below")
        n = len(nn["D"])
        L.append(f"{n:,} survey shapes passed phase A; {int(fin.sum()):,} have a neighbour with another "
                 f"thing_id; {len(nn['same']):,} same-thing pairs; {int(np.sum(nn['nn_d'][fin] == 0)):,} "
                 f"cross-thing NN distances are exactly 0.")
        L += ["", f"![dedup nn]({f})", ""]
        tree = cKDTree(nn["D"])
        ts = sorted(set(DEDUP_TABLE_THRESHOLDS) | ({thr} if thr else set()))
        rows = []
        for t in ts:
            pairs = tree.query_pairs(r=t, output_type="ndarray")
            if len(pairs):
                dist = np.linalg.norm(nn["D"][pairs[:, 0]] - nn["D"][pairs[:, 1]], axis=1)
                pairs = pairs[dist < t]
            g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) \
                else sp.coo_matrix((n, n))
            _, lab = connected_components(g, directed=False)
            sizes = np.bincount(lab)
            k = int(np.sum(nn["nn_d"][fin] < t))
            rows.append([f"{t:g}" + (" (current)" if t == thr else ""), f"{k:,} ({_pct(k / n)})",
                         int(len(pairs)), int(sizes.max()), int(sizes[sizes > 1].sum())])
        L += ["Per threshold: shapes with a cross-thing neighbour below it; near pairs (any thing) below it; "
              "largest connected group from those pairs (thing links ignored); shapes in groups of size > 1.", ""]
        L += _table(["threshold", "shapes with cross-thing NN below", "near pairs", "largest group",
                     "shapes in groups"], rows)
        dr = self._dedup_projection()
        if dr:
            L += [dr, ""]
        L += self._pair_figure()
        L += self._threshold_pair_figure()
        return L

    def _dedup_projection(self) -> str | None:
        thr = self.cfg.get("dedup", {}).get("near_threshold")
        if self.desc is None or thr is None or not {"hashes", "n_faces"} <= set(self.desc):
            return None
        from moe_data import dedup
        th = [None if (isinstance(t, float) and np.isnan(t)) else t for t in self.desc["thing"].tolist()]
        recs = dedup.dedup_records(self.desc["ids"].tolist(), self.desc["D"], self.desc["hashes"].tolist(), th,
                                   thr, self.desc["n_faces"].tolist())
        kept = sum(1 for r in recs.values() if r["keep"])
        groups = pd.Series([r["group_id"] for r in recs.values()]).value_counts()
        self.projection["thingi_dedup_kept"] = kept
        key = ("dedup", "near_threshold")
        if key in self.threshold_summary:
            self.threshold_summary[key] += (f"; dedup_records keeps {kept:,} of {len(recs):,}, largest split "
                                            f"group {int(groups.max())} shapes")
        return (f"dedup.dedup_records on the survey at near_threshold {thr:g}: {kept:,} of {len(recs):,} kept "
                f"({len(recs) - kept:,} rejected as duplicates); {len(groups):,} split groups, largest "
                f"{int(groups.max())} shapes (exact + near + same thing).")

    def _nn_pairs(self) -> list[tuple[int, int, float]]:
        """Unique cross-thing nearest-neighbour pairs (i, j, distance), ascending distance then id."""
        nn = self.nn
        idx = np.nonzero(np.isfinite(nn["nn_d"]))[0]
        seen, pairs = set(), []
        for i in idx[np.lexsort((nn["ids"][idx], nn["nn_d"][idx]))]:
            j = int(nn["nn_j"][i])
            key = (min(int(i), j), max(int(i), j))
            if key not in seen:
                seen.add(key)
                pairs.append((key[0], key[1], float(nn["nn_d"][i])))
        return pairs

    def _draw_pair(self, fig, nrows: int, ncols: int, slot: int, i: int, j: int, header: str) -> int:
        """Draw survey shapes i and j side by side in subplot slots slot, slot + 1 (same view and scale).
        Returns the number of missing meshes."""
        nn, mdir = self.nn, self.survey_meshes
        meshes = [_read_npz(mdir / f"{nn['ids'][x]}.npz", ("V", "F")) for x in (i, j)]
        ctr = [m["V"].mean(0) if m else np.zeros(3) for m in meshes]
        half = max((np.abs(m["V"] - c).max() for m, c in zip(meshes, ctr) if m), default=1.0)
        missing = 0
        for s, (x, m, c) in enumerate(zip((i, j), meshes, ctr)):
            ax = fig.add_subplot(nrows, ncols, slot + s, projection="3d")
            ax.set_axis_off()
            title = f"{nn['ids'][x]} (thing {nn['thing'][x]})"
            ax.set_title((header if s == 0 else "") + "\n" + title, fontsize=7)
            if not m:
                missing += 1
                ax.text2D(0.5, 0.5, "mesh missing", transform=ax.transAxes, ha="center")
                continue
            F = m["F"]
            if len(F) > PAIR_MAX_FACES:
                F = F[np.linspace(0, len(F) - 1, PAIR_MAX_FACES).astype(np.int64)]
            V = m["V"] - c
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, color="#9DB4D0", shade=True,
                            linewidth=0, antialiased=False)
            for setlim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                setlim(-half, half)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=-60)
        return missing

    def _meshes_ok(self) -> bool:
        return self.survey_meshes is not None and self.survey_meshes.is_dir()

    def _pair_figure(self) -> list[str]:
        pairs = self._nn_pairs()
        chosen = [[p for p in pairs if lo <= p[2] < hi][:PAIRS_PER_BIN] for lo, hi in DEDUP_PAIR_BINS]
        if not any(chosen):
            return ["Nearest-neighbour pair figure: no cross-thing pairs in the bins.", ""]
        if not self._meshes_ok():
            return [f"Nearest-neighbour pair figure: {NA} (normalised meshes not found at {self.survey_meshes}).",
                    ""]
        nb, nc = len(DEDUP_PAIR_BINS), 2 * PAIRS_PER_BIN
        fig = plt.figure(figsize=(3.0 * nc, 3.3 * nb))
        missing = 0
        for b, (bin_pairs, (lo, hi)) in enumerate(zip(chosen, DEDUP_PAIR_BINS)):
            if not bin_pairs:
                ax = fig.add_subplot(nb, nc, b * nc + 1)
                ax.axis("off")
                ax.text(0, 0.5, f"[{lo:g}, {hi:g}): no pairs", fontsize=9)
            for k, (i, j, dist) in enumerate(bin_pairs):
                missing += self._draw_pair(fig, nb, nc, b * nc + 2 * k + 1, i, j,
                                           f"d = {dist:.4g}  bin [{lo:g}, {hi:g})")
        fig.suptitle("Cross-thing nearest-neighbour pairs by descriptor distance (same view and scale per pair)",
                     fontsize=10)
        f = _save(fig, self.fig / "dedup_pairs.png")
        counts = ", ".join(f"[{lo:g}, {hi:g}): {len(c)}" for c, (lo, hi) in zip(chosen, DEDUP_PAIR_BINS))
        out = [f"Pairs shown per bin (lowest distances first): {counts}.", "", f"![dedup pairs]({f})", ""]
        if missing:
            out.insert(0, f"{missing} meshes missing from {self.survey_meshes}.")
        return out

    def _threshold_pair_figure(self) -> list[str]:
        """Cross-thing NN pairs just below and just above dedup.near_threshold, drawn side by side; saved as
        fig/dedup_threshold_pairs.png and a standalone copy <out>/dedup_threshold_pairs.png."""
        nn, thr = self.nn, self.cfg.get("dedup", {}).get("near_threshold")
        L = [f"#### Pairs around the near_threshold ({_f(thr)})", ""]
        if thr is None:
            return L + [f"{NA} (no dedup.near_threshold).", ""]
        below, above = threshold_pairs(self._nn_pairs(), thr, THRESHOLD_PAIRS_PER_SIDE)
        nf = np.asarray(self.desc["n_faces"]) if "n_faces" in self.desc else None
        rows = []
        for side, ps in (("below (merged)", below), ("above (kept apart)", above)):
            for k, (i, j, d) in enumerate(ps, 1):
                rows.append([side, k, f"`{nn['ids'][i]}`", nn["thing"][i], f"`{nn['ids'][j]}`", nn["thing"][j],
                             f"{d:.5f}", f"{_f(nf[i])} / {_f(nf[j])}" if nf is not None else NA])
        L += [f"The {THRESHOLD_PAIRS_PER_SIDE} highest cross-thing nearest-neighbour distances below the "
              f"threshold and the {THRESHOLD_PAIRS_PER_SIDE} lowest at or above it (survey descriptors).", ""]
        L += _table(["side", "rank", "id a", "thing a", "id b", "thing b", "distance", "faces a / b"], rows)
        if not rows:
            return L
        if not self._meshes_ok():
            return L + [f"Figure: {NA} (normalised meshes not found at {self.survey_meshes}).", ""]
        per_row = 2                                     # pairs per figure row
        sides = [("below", below), ("above", above)]
        nrows = sum(math.ceil(len(ps) / per_row) for _, ps in sides)
        nc = 2 * per_row
        fig = plt.figure(figsize=(3.2 * nc, 3.4 * nrows))
        r, missing = 0, 0
        for name, ps in sides:
            for k, (i, j, d) in enumerate(ps):
                row, col = r + k // per_row, k % per_row
                missing += self._draw_pair(fig, nrows, nc, row * nc + 2 * col + 1, i, j,
                                           f"{name.upper()} {thr:g}: d = {d:.5f}")
            r += math.ceil(len(ps) / per_row)
        fig.suptitle(f"Cross-thing NN pairs just below (top rows) and just above (bottom rows) dedup.near_threshold "
                     f"= {thr:g}; same view and scale per pair, <= {PAIR_MAX_FACES:,} faces drawn", fontsize=10)
        fig.savefig(self.out / "dedup_threshold_pairs.png", dpi=DPI, bbox_inches="tight")
        f = _save(fig, self.fig / "dedup_threshold_pairs.png")
        if missing:
            L += [f"{missing} meshes missing from {self.survey_meshes}.", ""]
        return L + [f"![dedup threshold pairs]({f})", ""]

    # ---- 3. timing and projection
    def timing(self) -> list[str]:
        L = ["## 3. Timing" + ("" if self.final else " and full-run projection"), ""]
        rows = []
        for s in self.stages:
            t = _num(self.df, f"time.{s}").dropna()
            if len(t):
                rows.append([s, len(t), _f(t.median()), _f(t.quantile(0.9)), _f(t.max()), _f(t.sum())])
        if rows:
            L.append("Per-stage wall time in this manifest (seconds, per shape that ran the stage):")
            L.append("")
            L += _table(["stage", "n", "median", "p90", "max", "sum"], rows)
            rows = []
            for s in self.stages:
                cells = []
                for g in self.groups:
                    t = _num(self.df, f"time.{s}")[self.df["_group"] == g].dropna()
                    cells.append(f"{_f(t.median())} / {_f(t.quantile(0.9))} / {_f(t.sum())} (n {len(t):,})"
                                 if len(t) else "-")
                if any(c != "-" for c in cells):
                    rows.append([s] + cells)
            L += ["Per group: median / p90 / sum seconds (n shapes that ran the stage):", ""]
            L += _table(["stage"] + self.groups, rows)
        else:
            L += [f"Stage timings: {NA}.", ""]
        if self.survey_df is not None:
            rows = []
            for s in PHASE_A:
                t = _num(self.survey_df, f"time.{s}").dropna()
                if len(t):
                    rows.append([s, len(t), _f(t.median()), _f(t.quantile(0.9)), _f(t.max())])
            if rows:
                L += ["Survey (full Thingi10K, phase A):", ""]
                L += _table(["stage", "n", "median", "p90", "max"], rows)
        if not self.final:
            L += self._projection()
        return L

    def _full_thingi_count(self) -> tuple[int | None, str]:
        if self.survey_df is not None:
            return len(self.survey_df), "survey phase_a.parquet rows"
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from moe_data import thingi
            return len(thingi.list_candidates(self.cfg, pilot=False)), "Thingi10K metadata (thingi.list_candidates)"
        except Exception as e:  # metadata cache missing: say so
            return None, f"{NA} ({type(e).__name__})"

    def _projection(self) -> list[str]:
        cfg, df = self.cfg, self.df
        timed = [s for s in self.stages if f"time.{s}" in df.columns
                 or (self.survey_df is not None and f"time.{s}" in self.survey_df.columns)]
        n_thingi, thingi_src = self._full_thingi_count()
        pc = cfg.get("procedural", {})
        full = {"thingi10k": n_thingi, "proc_blob": pc.get("n_blobs"), "proc_protrusion": pc.get("n_protrusion")}
        assumptions = [f"Thingi10K full count: {_f(n_thingi)} from {thingi_src}; procedural counts from config "
                       f"(n_blobs {pc.get('n_blobs')}, n_protrusion {pc.get('n_protrusion')})."]
        all_flow = {s["stage"]: s for s in stage_flow(df, self.stages)}

        def mean_time(g, s, frame=None):
            frame = df if frame is None else frame
            gl = group_labels(frame) if frame is not df else df["_group"]
            t = _num(frame, f"time.{s}")[gl == g].dropna()
            if len(t):
                return float(t.mean()), "group"
            t = _num(df, f"time.{s}").dropna()
            return (float(t.mean()), "pooled") if len(t) else (None, "none")

        reach: dict[str, dict[str, float]] = {}
        accepted: dict[str, float] = {}
        notes = set()
        for g, N in full.items():
            if N is None:
                continue
            sub = df[df["_group"] == g]
            flow = {s["stage"]: s for s in stage_flow(sub, self.stages)} if len(sub) else {}
            r: dict[str, float] = {}
            if g == "thingi10k" and self.survey_df is not None:
                sflow = {s["stage"]: s for s in stage_flow(self.survey_df, [s for s in self.stages if s in PHASE_A])}
                for s in PHASE_A:
                    if s in sflow:
                        r[s] = float(sflow[s]["in"])
                base = self.projection.get("thingi_dedup_kept")
                if base is None:
                    base = float(sflow["normalize"]["out"]) if "normalize" in sflow else None
                    notes.add("Thingi10K dedup survivors: dedup not computable, phase-A survivors used.")
                else:
                    notes.add("Thingi10K phase A reach from the survey; dedup survivors from dedup_records on "
                              "the survey descriptors.")
                use = flow if flow.get("remesh", {}).get("in") else all_flow
                if use is all_flow:
                    notes.add("No pilot Thingi10K shape reached remesh: Thingi10K phase-B pass rates use the "
                              "pooled pilot rates of all sources.")
                denom = use.get("remesh", {}).get("in", 0)
                after_dedup = [s for s in self.stages if s not in PHASE_A and s != "dedup"]
                for s in after_dedup:
                    if s in use and denom and base is not None:
                        r[s] = base * use[s]["in"] / denom
                last = use[self.stages[-1]]["out"] if denom else 0
                accepted[g] = base * last / denom if denom and base is not None else float("nan")
            else:
                if not len(sub):
                    notes.add(f"{g}: not in this manifest, projection {NA}.")
                    continue
                for s in self.stages:
                    r[s] = N * flow[s]["in"] / len(sub)
                accepted[g] = N * flow[self.stages[-1]]["out"] / len(sub)
                notes.add("Procedural and (without survey) Thingi10K reach = full count x pilot fraction "
                          "reaching each stage.")
            reach[g] = r

        workers = int(cfg.get("run", {}).get("max_workers", 1))
        rows, total, cached = [], 0.0, 0.0
        for g, r in reach.items():
            for s in timed:
                cnt = r.get(s)
                if cnt is None:
                    continue
                frame = self.survey_df if (g == "thingi10k" and s in PHASE_A and self.survey_df is not None) else None
                mt, how = mean_time(g, s, frame)
                if mt is None:
                    continue
                cpu = mt * cnt
                total += cpu
                if g == "thingi10k" and s in PHASE_A and self.survey_df is not None:
                    cached += cpu
                rows.append([g, s, _f(cnt), _f(mt), how + (" (survey)" if frame is not None else ""), _f(cpu / 3600)])
        L = ["### Projected full run", ""]
        if not rows:
            return L + [f"Projection: {NA}.", ""]
        L += _table(["group", "stage", "projected shapes", "mean s", "time source", "CPU hours"], rows)
        wall = total / workers / 3600
        L.append(f"Projected wall time: {_f(wall)} h = sum(mean stage time x projected count) / max_workers "
                 f"({workers})" + (f"; {_f((total - cached) / workers / 3600)} h if the Thingi10K phase A cached "
                                   f"by the survey is reused." if cached else "."))
        acc = sum(v for v in accepted.values() if np.isfinite(v))
        L.append("")
        L.append("Projected accepted: " + ", ".join(f"{g} {_f(v)}" for g, v in accepted.items())
                 + f"; total {_f(acc)}.")
        # disk
        sizes = []
        for sid in df.loc[self.ok, "id"]:
            p = self.processed / f"{sid}.npz"
            if p.exists():
                sizes.append(p.stat().st_size)
        work_sizes = scan_work(self.work)
        n_cand = sum(N for N in full.values() if N is not None)
        L.append("")
        if sizes:
            proc_gb = np.mean(sizes) * acc / 1e9
            L.append(f"Projected processed disk: mean file {np.mean(sizes) / 1e6:.2f} MB x {_f(acc)} = "
                     f"{_f(proc_gb)} GB.")
        else:
            proc_gb = None
            L.append(f"Projected processed disk: {NA} (no processed files found).")
        if work_sizes:
            wa = [work_sizes[s] for s in df.loc[self.ok, "id"] if s in work_sizes]
            wr = [work_sizes[s] for s in df.loc[~self.ok, "id"] if s in work_sizes]
            wa_m = float(np.mean(wa)) if wa else 0.0
            wr_m = float(np.mean(wr)) if wr else 0.0
            work_gb = (wa_m * acc + wr_m * max(n_cand - acc, 0)) / 1e9
            L.append(f"Projected work cache: {wa_m / 1e6:.2f} MB per accepted shape x {_f(acc)} + "
                     f"{wr_m / 1e6:.3f} MB per rejected candidate x {_f(max(n_cand - acc, 0))} = {_f(work_gb)} GB "
                     f"(work dir {self.work}).")
        else:
            work_gb = None
            L.append(f"Projected work cache: {NA} (work dir {self.work} not found).")
        self.projection.update(wall_h=wall, accepted=acc, proc_gb=proc_gb, work_gb=work_gb, workers=workers)
        L += ["", "Assumptions:", ""] + [f"- {a}" for a in assumptions + sorted(notes)]
        L += ["- Perfect parallel scaling over max_workers; dedup (dataset level) and assembly are not timed.",
              "- Mean stage time per group from the pilot; a group with no timing for a stage uses the pooled "
              "mean.", "- Pass rates measured on the pilot carry over to the full set.", ""]
        return L

    # ---- 4. shapes
    def _pick_shapes(self) -> tuple[list[str], list[str]]:
        acc = self.df[self.ok].copy()
        acc["_rng"] = _num(acc, "psi.psi_range")
        acc = acc.sort_values(["_rng", "id"])

        def spread(pool: pd.DataFrame, k: int) -> list[str]:
            if k <= 0 or not len(pool):
                return []
            k = min(k, len(pool))
            idx = sorted(set(np.round(np.linspace(0, len(pool) - 1, k)).astype(int)))
            return pool["id"].iloc[idx].tolist()

        thingi = spread(acc[acc["_group"] == "thingi10k"], N_SHAPES_PER_SIDE)
        target = 2 * N_SHAPES_PER_SIDE - len(thingi)
        prot, blob = acc[acc["_group"] == "proc_protrusion"], acc[acc["_group"] == "proc_blob"]
        n_prot = min(len(prot), max(math.ceil(target / 2), target - len(blob)))
        proc = spread(blob, target - n_prot) + spread(prot, n_prot)
        proc = sorted(proc, key=lambda s: acc.set_index("id").loc[s, "_rng"])
        return thingi, proc

    def shapes(self) -> list[str]:
        L = ["## 4. Shapes", ""]
        thingi, proc = self._pick_shapes()
        if not thingi and not proc:
            return L + [f"Shape figures: {NA} (no accepted shapes).", ""]
        data = {}
        for sid in thingi + proc:
            d = _read_npz(self.processed / f"{sid}.npz", ("V", "F", "P", "psi", "area", "grid", "grid_channels"))
            if d and {"V", "F", "psi", "area", "grid", "grid_channels"} <= set(d):
                data[sid] = d
        missing = [s for s in thingi + proc if s not in data]
        if not data:
            return L + [f"Shape figures: {NA} (processed files missing).", ""]
        chans = [str(c) for c in next(iter(data.values()))["grid_channels"]]
        rel = {s: d["psi"] - np.sum(d["area"] * d["psi"]) / np.sum(d["area"]) for s, d in data.items()}
        allrel = np.concatenate(list(rel.values()))
        psi_lim = (np.percentile(allrel, 1), np.percentile(allrel, 99))
        lims, cmaps = [], []
        for c in range(len(chans)):
            v = np.concatenate([d["grid"][c].ravel() for d in data.values() if d["grid"].shape[0] > c])
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            if lo < 0 < hi and chans[c] not in ("psi", "psi_min"):
                m = max(-lo, hi)
                lims.append((-m, m))
                cmaps.append("RdBu_r")
            else:
                lims.append((lo, hi if hi > lo else lo + 1e-12))
                cmaps.append("viridis")
        figs = []
        for name, ids in (("shapes_thingi.png", [s for s in thingi if s in data]),
                          ("shapes_procedural.png", [s for s in proc if s in data])):
            if ids:
                figs.append(self._shape_figure(ids, data, rel, chans, psi_lim, lims, cmaps, self.fig / name))
        groups = self.df.set_index("id")["_group"]
        L.append("Chosen deterministically: accepted shapes sorted by psi range, evenly spaced picks; "
                 f"Thingi10K: {len(thingi)}, procedural: {len(proc)} "
                 f"({', '.join(f'{s} [{groups[s]}]' for s in proc)}). Left: surface coloured by psi minus its "
                 "area-weighted mean (shaded); right: stored grid channels (lat-long, row 0 = north). "
                 "Colour limits are the 1st-99th percentile over all shown shapes, shared across both figures.")
        L.append("")
        if missing:
            L += [f"Processed files missing for: {', '.join(missing)}.", ""]
        L += [f"![shapes]({f})" for f in figs] + [""]
        return L

    def _shape_figure(self, ids, data, rel, chans, psi_lim, lims, cmaps, path) -> str:
        nc = 1 + len(chans)
        fig = plt.figure(figsize=(2.6 + 2.5 * len(chans), 1.55 * len(ids) + 0.9))
        gs = fig.add_gridspec(len(ids) + 1, nc, height_ratios=[1] * len(ids) + [0.08],
                              width_ratios=[1.0] + [1.6] * len(chans), hspace=0.25, wspace=0.12)
        norm3 = mcolors.Normalize(*psi_lim)
        cmap3 = plt.get_cmap("viridis")
        light = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])
        ims = [None] * len(chans)
        for r, sid in enumerate(ids):
            d = data[sid]
            V, F = d["V"], d["F"]
            ax = fig.add_subplot(gs[r, 0], projection="3d")
            fpsi = rel[sid][F].mean(axis=1)
            T = V[F]
            nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
            shade = 0.45 + 0.55 * np.abs(nrm @ light)
            col = cmap3(norm3(fpsi))
            col[:, :3] *= shade[:, None]
            ax.add_collection3d(Poly3DCollection(T, facecolors=col, linewidths=0, antialiased=False))
            c, half = V.mean(0), np.abs(V - V.mean(0)).max()
            ax.set_xlim(c[0] - half, c[0] + half)
            ax.set_ylim(c[1] - half, c[1] + half)
            ax.set_zlim(c[2] - half, c[2] + half)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=-60)
            ax.set_axis_off()
            ax.text2D(0.0, 0.98, sid, transform=ax.transAxes, fontsize=7, va="top")
            for k, ch in enumerate(chans):
                a2 = fig.add_subplot(gs[r, k + 1])
                if k < d["grid"].shape[0]:
                    ims[k] = a2.imshow(d["grid"][k], extent=[-180, 180, -90, 90], cmap=cmaps[k],
                                       vmin=lims[k][0], vmax=lims[k][1], aspect="auto", interpolation="nearest")
                a2.set_xticks([-180, 0, 180] if r == len(ids) - 1 else [])
                a2.set_yticks([-90, 0, 90] if k == 0 else [])
                a2.tick_params(labelsize=6)
                if r == 0:
                    a2.set_title(ch, fontsize=8)
        cax = fig.add_subplot(gs[-1, 0])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm3, cmap=cmap3), cax=cax, orientation="horizontal")
        cax.set_xlabel("psi - mean (surface)", fontsize=7)
        cax.tick_params(labelsize=6)
        for k, im in enumerate(ims):
            cax = fig.add_subplot(gs[-1, k + 1])
            if im is not None:
                fig.colorbar(im, cax=cax, orientation="horizontal")
            cax.tick_params(labelsize=6)
        return _save(fig, path)

    # ---- 5. validation
    def validation(self) -> list[str]:
        L = ["## 5. Validation summary (accepted shapes)", ""]
        acc = self.df[self.ok]
        if not len(acc):
            return L + ["No accepted shapes.", ""]
        v, c = self.cfg.get("validate", {}), self.cfg.get("curvature", {})
        checks = [
            ("centering residual |r|", "validate_sphere.centering_residual", "abs", v.get("centering_tol")),
            ("coverage error |area/4pi - 1|", "validate_sphere.coverage_err", "abs", v.get("coverage_tol")),
            ("max norm error ||p|-1|", "validate_sphere.max_norm_err", "abs", v.get("unit_norm_tol")),
            ("Gauss-Bonnet error", "curvature.gauss_bonnet_err", "abs", c.get("gauss_bonnet_tol")),
            ("grid mass error", "grid.mass_error", "abs", self.cfg.get("grid", {}).get("mass_tol")),
            ("min sphere edge", "validate_sphere.min_sphere_edge", "min", v.get("min_sphere_edge")),
            ("flipped triangles", "validate_sphere.n_flipped", "zero", 0),
            ("degenerate sphere triangles", "validate_sphere.n_degenerate", "zero", 0),
            ("non-finite P rows", "validate_sphere.n_nonfinite_P", "zero", 0),
            ("unlocated grid cells", "grid.n_unlocated", "zero", 0),
            ("max |psi - psi_lsq|", "psi.psi_vs_lsq_max_abs", "abs", None),
            ("max |psi - psi_lsq| outside corners", "psi.psi_vs_lsq_nocorner_max_abs", "abs", None),
        ]

        def check(frame: pd.DataFrame, col: str, kind: str, tol) -> tuple[str, str]:
            n, x = len(frame), _num(frame, col)
            if not x.notna().any():
                return NA, NA
            stat = ("min " + _f(x.min())) if kind == "min" else ("max " + _f(x.abs().max()))
            if tol is None:
                return stat, "no automated check"
            ok = (x >= tol) if kind == "min" else ((x == 0) if kind == "zero" else (x.abs() <= tol))
            return stat, f"{int(ok.sum())}/{n}"

        def flag(frame: pd.DataFrame, col: str) -> tuple[str, str]:
            if col not in frame.columns:
                return NA, NA
            return "-", f"{int(_flag_true(frame, col).sum())}/{len(frame)}"

        flags = (("vertex correspondence", "validate_sphere.correspondence_ok"),
                 ("view fraction in (0.05, 0.95) (informational)", "visibility.vis_view_frac_ok"))
        rows = []
        for name, col, kind, tol in checks:
            stat, passing = check(acc, col, kind, tol)
            rows.append([name, stat, _f(tol) if tol is not None else "-", passing])
        for name, col in flags:
            st, p = flag(acc, col)
            rows.append([name, st, "True", p])
        L += _table(["quantity", "worst over accepted", "tolerance (config)", "passing"], rows)
        rows = []
        subs = {g: acc[acc["_group"] == g] for g in self.groups if (acc["_group"] == g).any()}
        for name, col, kind, tol in checks:
            rows.append([name] + ["{} ({})".format(*check(s, col, kind, tol)) for s in subs.values()])
        for name, col in flags:
            rows.append([name] + [flag(s, col)[1] for s in subs.values()])
        L += ["Per group: worst value (passing / accepted in the group):", ""]
        L += _table(["quantity"] + list(subs), rows)
        return L

    # ---- 6. splits and dataset size
    def _split_frame(self) -> pd.DataFrame:
        acc = self.df[self.ok].copy()
        acc["_split"] = acc["split"].fillna("").replace("", "(none)") if "split" in acc.columns else "(none)"
        acc["_bytes"] = [self._file_bytes(s) for s in acc["id"]]
        return acc

    def _file_bytes(self, sid: str) -> float:
        p = self.processed / f"{sid}.npz"
        try:
            return float(p.stat().st_size)
        except OSError:
            return float("nan")

    def split_table(self, with_bytes: bool = True) -> list[str]:
        """Accepted shapes per split (rows, every configured label even when empty) and group (columns)."""
        acc = self._split_frame()
        labels = list(SPLIT_LABELS) + sorted(set(acc["_split"]) - set(SPLIT_LABELS))
        rows = []
        for lab in labels + ["total"]:
            sub = acc if lab == "total" else acc[acc["_split"] == lab]
            cells = [int((sub["_group"] == g).sum()) for g in self.groups]
            row = [lab] + cells + [len(sub)]
            if with_bytes:
                row.append(_f(np.nansum(sub["_bytes"]) / 1e9, 3) if len(sub) else "0")
            rows.append(row)
        return _table(["split"] + self.groups + ["total"] + (["processed GB"] if with_bytes else []), rows)

    def splits(self) -> list[str]:
        L = ["## 6. Splits and dataset size", ""]
        if "split" not in self.df.columns:
            return L + [f"Splits: {NA}.", ""]
        L += ["Accepted shapes per split and group (all split labels listed; a label the manifest's run did "
              "not produce shows 0, e.g. protrusion_test_inrange in manifests from before it existed), with "
              "the size of their processed files:", ""]
        L += self.split_table()
        acc = self._split_frame()
        rows = []
        for lab in sorted(set(acc["_split"])):
            for g in self.groups:
                sub = acc[(acc["_split"] == lab) & (acc["_group"] == g)]
                if len(sub):
                    rows.append([lab, g, len(sub), _f(np.nansum(sub["_bytes"]) / 1e6, 4),
                                 _f(np.nanmean(sub["_bytes"]) / 1e6) if sub["_bytes"].notna().any() else NA])
        L += ["Processed size per split and group:", ""]
        L += _table(["split", "group", "shapes", "total MB", "mean MB per shape"], rows)
        return L

    # ---- 7. psi
    def psi_section(self) -> list[str]:
        pc = self.cfg.get("protrusion", {})
        L = ["## 7. psi distributions", "",
             f"Per shape (all shapes where psi was computed). Ranges are max - min of psi relative to its "
             f"area-weighted mean; 'outside corners' drops vertices within corners.exclude_radius = "
             f"{self.cfg.get('corners', {}).get('exclude_radius')} of a corner; area share low = area with "
             f"psi_rel < psi_low = {pc.get('psi_low')}; corner area share = area of the excluded vertices. "
             f"protrusion.range_source = {pc.get('range_source', 'psi_range')}.", ""]
        specs = [("psi range", "psi.psi_range"), ("psi range outside corners", "psi.psi_nocorner_range"),
                 ("area share low", "psi.psi_area_share_low"),
                 ("area share low outside corners", "psi.psi_nocorner_area_share_low"),
                 ("corner area share", "psi.corner_area_share")]
        L += self._dist_table([(lab, self._grouped(col)) for lab, col in specs])
        rt = pc.get("range_threshold")
        cas = self._grouped("psi.corner_area_share")
        if len(cas["all"]):
            self.threshold_summary[("corners", "exclude_radius")] = "corner area share " + "; ".join(
                f"{g} median {_f(x.median())}, max {_f(x.max())}" for g, x in cas.items() if len(x)) \
                + " (section 7)"
        if rt is not None:
            th = self.df[self.df["_group"] == "thingi10k"]
            a, b = _num(th, "psi.psi_range"), _num(th, "psi.psi_nocorner_range")
            if a.notna().any() or b.notna().any():
                self.threshold_summary[("protrusion", "range_source")] = (
                    f"Thingi10K shapes over range_threshold {rt:g}: {int((a > rt).sum())} by psi_range, "
                    f"{int((b > rt).sum())} by psi_nocorner_range (section 7)")
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        _hist_panel(axs[0, 0], _by_group(self.df, "psi.psi_range"), rt, [], "above", False, "psi range",
                    "psi range (all vertices)", verb="above")
        _hist_panel(axs[0, 1], _by_group(self.df, "psi.psi_nocorner_range"), rt, [], "above", False,
                    "psi range outside corners", "psi range (corners excluded)", verb="above")
        _hist_panel(axs[1, 0], _by_group(self.df, "psi.psi_area_share_low"), None, [], "above", False,
                    "area share with psi_rel < psi_low", "area share low")
        _hist_panel(axs[1, 1], _by_group(self.df, "psi.corner_area_share"), None, [], "above", False,
                    "area share of corner-excluded vertices", "corner area share")
        L += [f"![psi distributions]({_save(fig, self.fig / 'psi_distributions.png')})", ""]
        return L

    # ---- 8. curvature
    def _sample_ids(self) -> dict[str, list[str]]:
        """Accepted ids per group, at most CURV_SAMPLE_PER_GROUP, evenly spaced over the sorted ids."""
        out = {}
        acc = self.df[self.ok]
        for g in self.groups:
            ids = sorted(acc.loc[acc["_group"] == g, "id"].astype(str))
            if len(ids) > CURV_SAMPLE_PER_GROUP:
                ids = [ids[i] for i in np.linspace(0, len(ids) - 1, CURV_SAMPLE_PER_GROUP).astype(int)]
            if ids:
                out[g] = ids
        return out

    def curvature_section(self) -> list[str]:
        L = ["## 8. Curvature distributions", ""]
        sample = self._sample_ids()
        recs = []
        for g, ids in sample.items():
            for sid in ids:
                d = _read_npz(self.processed / f"{sid}.npz", ("K_int", "K_pt", "H"))
                if not d or not {"K_int", "K_pt", "H"} <= set(d):
                    continue
                kp, h = d["K_pt"][np.isfinite(d["K_pt"])], d["H"][np.isfinite(d["H"])]
                recs.append({"id": sid, "_group": g, "K_int_maxabs": float(np.nanmax(np.abs(d["K_int"]))),
                             **{f"K_pt_{q}": float(np.quantile(kp, p)) if kp.size else np.nan
                                for q, p in (("q01", 0.01), ("q50", 0.5), ("q99", 0.99))},
                             **{f"H_{q}": float(np.quantile(h, p)) if h.size else np.nan
                                for q, p in (("q01", 0.01), ("q50", 0.5), ("q99", 0.99))}})
        n_acc = {g: int(((self.df["_group"] == g) & self.ok).sum()) for g in sample}
        if not recs:
            return L + [f"Curvature from processed files: {NA} (no processed files).", ""]
        cv = pd.DataFrame(recs)
        L += ["Per-shape statistics read from the processed files (" + ", ".join(
            f"{g}: {int((cv['_group'] == g).sum())} of {n_acc[g]} accepted" for g in sample)
              + f"; at most {CURV_SAMPLE_PER_GROUP} per group, evenly spaced over sorted ids), then their "
              "distribution over shapes. K_int is the per-vertex integrated Gauss curvature (angle defect), "
              "K_pt and H the pointwise Gauss and mean curvature on the unit-area mesh.", ""]
        specs = [("max |K_int|", "K_int_maxabs"), ("K_pt p1", "K_pt_q01"), ("K_pt median", "K_pt_q50"),
                 ("K_pt p99", "K_pt_q99"), ("H p1", "H_q01"), ("H median", "H_q50"), ("H p99", "H_q99")]
        L += self._dist_table([(lab, self._grouped(col, cv)) for lab, col in specs])
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        bg = {g: cv.loc[cv["_group"] == g, c].to_numpy() for g in sample for c in ["K_int_maxabs"]}
        _hist_panel(axs[0], bg, None, [], "above", True, "max |K_int| per shape (rad, log)", "max |K_int|")
        _hist_panel(axs[1], {g: cv.loc[cv["_group"] == g, "K_pt_q99"].to_numpy() for g in sample}, None, [],
                    "above", True, "K_pt p99 per shape (log; <= 0 not shown)", "K_pt p99")
        _hist_panel(axs[2], {g: cv.loc[cv["_group"] == g, "H_q50"].to_numpy() for g in sample}, None, [],
                    "above", False, "H median per shape", "H median")
        L += [f"![curvature]({_save(fig, self.fig / 'curvature.png')})", ""]
        return L

    # ---- 9. psi vs psi_lsq
    def lsq_section(self) -> list[str]:
        L = ["## 9. psi vs psi_lsq", "",
             "max |psi - psi_lsq| per shape over all vertices and outside corners (both relative to their "
             "area-weighted means), and the edge least-squares residual:", ""]
        L += self._dist_table([("max |psi - psi_lsq|", self._grouped("psi.psi_vs_lsq_max_abs")),
                               ("max |psi - psi_lsq| outside corners",
                                self._grouped("psi.psi_vs_lsq_nocorner_max_abs")),
                               ("LSQ residual RMS", self._grouped("psi.lsq_residual_rms"))])
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        _hist_panel(axs[0], _by_group(self.df, "psi.psi_vs_lsq_max_abs"), None, [], "above", True,
                    "max |psi - psi_lsq| (log)", "psi vs psi_lsq, all vertices")
        _hist_panel(axs[1], _by_group(self.df, "psi.psi_vs_lsq_nocorner_max_abs"), None, [], "above", True,
                    "max |psi - psi_lsq| outside corners (log)", "psi vs psi_lsq, corners excluded")
        _hist_panel(axs[2], _by_group(self.df, "psi.lsq_residual_rms"), None, [], "above", True,
                    "edge LSQ residual RMS (log)", "psi_lsq residual")
        L += [f"![psi vs lsq]({_save(fig, self.fig / 'psi_lsq.png')})", ""]
        return L

    # ---- 10. grid
    def grid_section(self) -> list[str]:
        tol = self.cfg.get("grid", {}).get("mass_tol")
        L = ["## 10. Grid mass error", "",
             f"|total curvature on the grid - 4 pi| per shape (grid.mass_tol = {_f(tol)}), and grid locator "
             "counts:", ""]
        L += self._dist_table([("grid mass error", self._grouped("grid.mass_error", absval=True)),
                               ("unlocated cells", self._grouped("grid.n_unlocated")),
                               ("locator fallbacks", self._grouped("grid.n_locator_fallback"))])
        x = _num(self.df, "grid.mass_error").abs()
        if tol is not None and x.notna().any():
            over = self.df.loc[x > tol, "id"].tolist()
            L += [f"Shapes over mass_tol: {len(over)}" + (f" ({', '.join(over[:20])}"
                                                          + (", ..." if len(over) > 20 else "") + ")" if over else "")
                  + ".", ""]
        return L

    # ---- 11. disk
    def disk_section(self) -> list[str]:
        L = ["## 11. Disk use", ""]
        acc = self._split_frame()
        rows = []
        for g in self.groups + ["all"]:
            sub = acc if g == "all" else acc[acc["_group"] == g]
            have = sub["_bytes"].notna()
            rows.append([g, len(sub), int(have.sum()), _f(np.nansum(sub["_bytes"]) / 1e9, 4),
                         _f(np.nanmean(sub["_bytes"]) / 1e6) if have.any() else NA,
                         _f(np.nanmax(sub["_bytes"]) / 1e6) if have.any() else NA])
        L += [f"Processed files under {self.processed} (accepted shapes):", ""]
        L += _table(["group", "accepted", "files found", "total GB", "mean MB", "max MB"], rows)
        tot, n = dir_bytes(self.processed)
        L += [f"Processed directory total: {_f(tot / 1e9, 4)} GB in {n:,} files.", ""]
        if self.work is None or not self.work.is_dir():
            return L + [f"Work cache: {NA} (work dir {self.work} not found).", ""]
        rows, total, files = [], 0, 0
        for d in sorted(p for p in self.work.iterdir() if p.is_dir()):
            b, k = dir_bytes(d)
            total, files = total + b, files + k
            rows.append([d.name, f"{k:,}", _f(b / 1e9, 4)])
        L += [f"Work cache {self.work}: {_f(total / 1e9, 4)} GB in {files:,} files.", ""]
        L += _table(["stage dir", "files", "GB"], rows)
        return L

    # ---- 12. stretch check (optional input)
    def stretch_section(self) -> list[str]:
        L = ["## 12. Stretch check", ""]
        if self.stretch is None or not self.stretch.is_file():
            return L + [f"{NA} (no stretch summary at {self.stretch}).", ""]
        return L + [f"Verbatim from {self.stretch}:", "", self.stretch.read_text().rstrip(), ""]

    # ---- final-run lead sections
    def _top(self, col: str, k: int, ascending: bool = False, frame: pd.DataFrame | None = None,
             extra: tuple[str, ...] = ()) -> list[list]:
        frame = self.df[self.ok] if frame is None else frame
        if col not in frame.columns:
            return []
        x = _num(frame, col)
        sub = frame.assign(_v=x)[x.notna()].sort_values(["_v", "id"], ascending=[ascending, True]).head(k)
        cols = {c: _num(sub, c) if c in sub.columns else pd.Series(np.nan, index=sub.index) for c in extra}
        return [[f"`{r['id']}`", r["_group"], _f(r["_v"], 4)] + [_f(cols[c][i], 4) for c in extra]
                for i, r in sub.iterrows()]

    def final_lead(self) -> list[str]:
        df = self.df
        L = ["## Final dataset", "", "Accepted shapes per split and source group:", ""]
        L += self.split_table()
        # folded maps
        fold = folded_mask(df)
        L += ["## Folded sphere maps (rejected at validate_sphere, kept in the manifest)", ""]
        cnt = {g: int((fold & (df["_group"] == g)).sum()) for g in self.groups}
        reached = {g: int(((df["_group"] == g) & _num(df, "validate_sphere.n_flipped").notna()).sum())
                   for g in self.groups}
        L += _table(["group", "folded", "validated (reached validate_sphere)", "folded share"],
                    [[g, cnt[g], reached[g], _pct(cnt[g] / reached[g]) if reached[g] else NA] for g in self.groups]
                    + [["all", int(fold.sum()), sum(reached.values()),
                        _pct(fold.sum() / sum(reached.values())) if sum(reached.values()) else NA]])
        fsub = df[fold].assign(_nf=_num(df, "validate_sphere.n_flipped")[fold]).sort_values(
            ["_nf", "id"], ascending=[False, True])
        if len(fsub):
            L += ["Folded shapes (filter with `sphere_folded`), most flipped triangles first:", ""]
            L += _table(["id", "group", "n_flipped", "min orientation"],
                        [[f"`{r['id']}`", r["_group"], _f(r["_nf"]), _f(_num(fsub, "validate_sphere.min_orient")[i])]
                         for i, r in fsub.iterrows()])
        if "sphere_collapsed" in df.columns:
            col = df["sphere_collapsed"].fillna(False).astype(bool)
            L += ["", "Flipped-triangle rejections whose shortest sphere edge is at the float64 limit are flagged "
                  "`sphere_collapsed` instead (compression too strong to represent, not a fold):", ""]
            L += _table(["group", "collapsed"], [[g, int((col & (df["_group"] == g)).sum())] for g in self.groups]
                        + [["all", int(col.sum())]])
        L += ["## Rejection reasons", ""] + self.reasons_ranked()
        L += ["## Items to inspect by hand", ""] + self.inspect_items()
        return L

    def inspect_items(self) -> list[str]:
        rsrc = self.cfg.get("protrusion", {}).get("range_source", "psi_range")
        acc = self.df[self.ok]
        L = []
        for g in self.groups:
            sub = acc[acc["_group"] == g]
            if not len(sub):
                continue
            ext = ("psi.psi_range", "psi.psi_nocorner_range")
            L += [f"Extremes of {rsrc} among accepted {g} (lowest, then highest {N_INSPECT}):", ""]
            low = self._top(f"psi.{rsrc}", N_INSPECT, True, sub, ext)
            high = [r for r in self._top(f"psi.{rsrc}", N_INSPECT, False, sub, ext)
                    if r[0] not in {x[0] for x in low}]          # small groups: no shape listed twice
            L += _table(["id", "group", rsrc, "psi range", "outside corners"], low + high)
        for title, col, extra in (
                ("Largest remesh Hausdorff among accepted", "remesh.hausdorff_sym", ()),
                ("Largest corner area share among accepted", "psi.corner_area_share", ("corners.n_corners",)),
                ("Largest |psi - psi_lsq| outside corners among accepted", "psi.psi_vs_lsq_nocorner_max_abs",
                 ("psi.psi_vs_lsq_max_abs",)),
                ("Largest grid mass error among accepted", "grid.mass_error", ())):
            rows = self._top(col, N_INSPECT, False, None, extra)
            L += [f"{title} (top {N_INSPECT}):", ""]
            L += _table(["id", "group", col] + list(extra), rows) if rows else [f"{NA}.", ""]
        if "dup_group" in self.df.columns:
            dg = self.df[self.df["dup_group"].notna()]
            sizes = dg["dup_group"].value_counts()
            big = sizes[sizes >= BIG_DUP_GROUP]
            L += [f"Dedup groups with >= {BIG_DUP_GROUP} members: {len(big)} (largest {int(sizes.max()) if len(sizes) else 0}).",
                  ""]
            rows = []
            for gid, n in big.items():
                mem = dg[dg["dup_group"] == gid].sort_values("id")
                kept = mem.loc[mem["ok"].astype(bool), "id"].tolist()
                reason = ", ".join(sorted(set(mem["dup_group_reason"].dropna().astype(str)))) \
                    if "dup_group_reason" in mem.columns else NA
                ids = mem["id"].tolist()
                rows.append([f"`{gid}`", int(n), ", ".join(kept) or "-", reason,
                             ", ".join(ids[:10]) + (", ..." if len(ids) > 10 else "")])
            if rows:
                L += _table(["group", "size", "accepted", "reason", "members"], rows)
        else:
            L += [f"Dedup groups: {NA} (no dup_group column).", ""]
        return L

    # ---- decisions (written last, placed first)
    def decisions_section(self) -> list[str]:
        L = ["## Decisions needed", ""]
        k = 1
        proposed = parse_proposed(self.config_path)
        for section, key in proposed:
            keys = [key] if key != "*" else list(self.cfg.get(section, {}))
            for kk in keys:
                val = self.cfg.get(section, {}).get(kk)
                s = self.threshold_summary.get((section, kk), "no distribution in this report")
                L.append(f"{k}. Confirm PROPOSED `{section}.{kk}` = {val}: {s}.")
                k += 1
        if self.final:                  # the full run is done: only the PROPOSED thresholds remain open
            if k == 1:
                L.append("No PROPOSED thresholds remain in the config.")
            return L + [""]
        lsq = _num(self.df, "psi.psi_vs_lsq_max_abs")
        L.append(f"{k}. Choose `grid.psi_source` (now `{self.cfg.get('grid', {}).get('psi_source')}`): max |psi - "
                 f"psi_lsq| over shapes " + (", ".join(
                     f"{g} median {_f(v.median())} / max {_f(v.max())}"
                     for g, v in lsq.groupby(self.df['_group']) if v.notna().any()) or NA) + " (section 9).")
        k += 1
        pp = self.cfg.get("split", {})
        prot = self._protrusion_params()
        if len(prot):
            n_len = int((prot["_max_len"] >= pp.get("holdout_length_min", np.inf)).sum())
            n_lr = int((prot["_max_lr"] > pp.get("holdout_lr_min", np.inf)).sum())
            L.append(f"{k}. Protrusion holdout rule (now `holdout_by: {pp.get('holdout_by')}`): of {len(prot)} "
                     f"protrusion shapes, {n_len} held out by length >= {pp.get('holdout_length_min')}, {n_lr} by "
                     f"L/r > {pp.get('holdout_lr_min')}.")
        else:
            L.append(f"{k}. Protrusion holdout rule: {NA} (no protrusion shapes with parameters).")
        k += 1
        if self.flags:
            L.append(f"{k}. Stages rejecting > 50% of their input (check: bug or threshold?): "
                     + "; ".join(self.flags) + ".")
        else:
            L.append(f"{k}. No stage rejects > 50% of its input (Thingi10K topology exempt).")
        k += 1
        p = self.projection
        if "wall_h" in p:
            L.append(f"{k}. Approve the full run: projected {_f(p['wall_h'])} h wall time on {p['workers']} workers, "
                     f"{_f(p['accepted'])} accepted shapes, processed {_f(p.get('proc_gb'))} GB + work cache "
                     f"{_f(p.get('work_gb'))} GB (section 3).")
        else:
            L.append(f"{k}. Full-run projection: {NA}.")
        L += ["", "Open items this report does not measure: rotation "
              "throughput vs training step rate, K_pt bias at irregular vertices.", ""]
        return L

    def build(self) -> str:
        body = []
        for part in (self.counts, self.thresholds, self.timing, self.shapes, self.validation, self.splits,
                     self.psi_section, self.curvature_section, self.lsq_section, self.grid_section,
                     self.disk_section, self.stretch_section):
            body += part()
        title = "# Final dataset report (full run)" if self.final else "# Dataset pipeline report"
        head = [title, "",
                f"Manifest: {len(self.df):,} candidates, {int(self.ok.sum()):,} accepted. Survey: "
                + (f"{len(self.survey_df):,} Thingi10K files" if self.survey_df is not None else NA) + ".", ""]
        lead = self.final_lead() if self.final else []
        return "\n".join(head + lead + self.decisions_section() + body) + "\n"


def _default_work(cfg: dict, processed: Path) -> Path | None:
    cands = [Path(cfg.get("paths", {}).get("work", "")) / processed.name, processed.parent.parent / "work" / processed.name]
    return next((c for c in cands if c.is_dir()), None)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--survey", default=None, help="dir with phase_a.parquet and descriptors.npz")
    ap.add_argument("--work", default=None, help="work cache of this run (default: paths.work/<processed name>)")
    ap.add_argument("--survey-meshes", default=None,
                    help="normalised survey meshes (default: paths.work/full/normalize)")
    ap.add_argument("--final", choices=("auto", "yes", "no"), default="auto",
                    help="final-report layout; auto = manifest is paths.manifest or processed dir is .../full")
    ap.add_argument("--stretch", default=None,
                    help="stretch summary to include (default: paths.reports/<processed name>/stretch/"
                         "STRETCH.md)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    processed = Path(args.processed)
    work = Path(args.work) if args.work else _default_work(cfg, processed)
    meshes = Path(args.survey_meshes) if args.survey_meshes else Path(cfg["paths"]["work"]) / "full" / "normalize"
    survey_df, desc = load_survey(args.survey)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = is_final_run(cfg, args.manifest, processed) if args.final == "auto" else args.final == "yes"
    stretch = Path(args.stretch) if args.stretch else \
        Path(cfg.get("paths", {}).get("reports", "data/reports")) / processed.name / "stretch" / "STRETCH.md"
    rep = Report(cfg, args.config, pd.read_parquet(args.manifest), processed, out, survey_df, desc, work, meshes,
                 final=final, stretch=stretch)
    text = rep.build()
    (out / "REPORT.md").write_text(text)
    print(json.dumps({"report": str(out / "REPORT.md"), "figures": sorted(p.name for p in (out / "fig").glob("*.png"))}))


if __name__ == "__main__":
    main()
