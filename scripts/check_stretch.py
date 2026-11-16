"""Measurement only: closed-form cone scale factor u vs the mesh Poisson solve on accepted shapes.

For each picked shape, two cone layouts ((a) 8 farthest-point cones on the sphere, theta = pi/2; (b) 4 cones at
the lowest-psi vertices outside the corner mask, mutually >= --min-sep apart on the surface, theta = pi) and both
psi estimators (psi, psi_lsq), runs moe_data.stretch.compare and writes stretch.csv, STRETCH.md and a figure.

Usage (pilot):
  uv run python scripts/check_stretch.py --manifest data/reports/pilot/manifest.parquet \
      --processed data/processed/pilot --out data/reports/pilot_stretch
Full run: --manifest data/manifest.parquet --processed data/processed/full --out data/reports/full_stretch
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from moe_data import geometry as geo
from moe_data import stretch as st
from moe_data.dataset import read_processed

GROUPS = ("Thingi10K", "proc_blob", "proc_protrusion")
SHARES = (0.4, 0.3, 0.3)
LAYOUTS = {"a_fps8": (8, np.pi / 2), "b_lowpsi4": (4, np.pi)}
PSI_KEYS = ("psi", "psi_lsq")
STAT_COLS = ["masked_max", "masked_p99", "masked_rms", "unmasked_max", "unmasked_p99", "unmasked_rms",
             "masked_area_share", "u_range", "u_dev_p99"]


def source_group(row) -> str:
    if row["source"] == "thingi10k":
        return "Thingi10K"
    fam = row.get("load.family")
    return {"blob": "proc_blob", "protrusion": "proc_protrusion"}.get(fam, f"proc_{fam}")


def pick_shapes(man: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic stratified sample of accepted shapes; quotas n*SHARES, shortfalls moved to other groups."""
    acc = man[man["ok"].astype(bool)].copy()
    acc["group"] = acc.apply(source_group, axis=1)
    pools = {g: sorted(acc.loc[acc["group"] == g, "id"]) for g in GROUPS}
    quota = dict(zip(GROUPS, np.floor(np.array(SHARES) * n).astype(int)))
    quota[GROUPS[0]] += n - sum(quota.values())
    short = sum(max(0, quota[g] - len(pools[g])) for g in GROUPS)
    quota = {g: min(quota[g], len(pools[g])) for g in GROUPS}
    while short > 0 and any(quota[g] < len(pools[g]) for g in GROUPS):
        for g in GROUPS:
            if short > 0 and quota[g] < len(pools[g]):
                quota[g] += 1
                short -= 1
    rng = np.random.default_rng(seed)
    ids = [i for g in GROUPS for i in rng.permutation(pools[g])[: quota[g]].tolist()]
    return acc.set_index("id").loc[ids, ["group"]].reset_index()


def layout_cones(name, d, r, min_sep):
    k, th = LAYOUTS[name]
    if name == "a_fps8":
        idx = st.farthest_point_cones(d["P"], k, start=int(np.argmax(d["P"][:, 2])))
    else:
        idx = st.compressed_cones(d["V"], d["psi"], k, min_sep, candidates=d["corner_dist"] >= r)
    return idx, np.full(k, th)


def run_shape(sid, group, d, r, min_sep):
    rows, fields = [], {}
    k_err = float(np.max(np.abs(d["K_int"] - geo.angle_defect(d["V"], d["F"]))))
    for lay in LAYOUTS:
        try:
            cones, theta = layout_cones(lay, d, r, min_sep)
        except ValueError as e:
            rows.append({"id": sid, "group": group, "layout": lay, "error": str(e)})
            continue
        for pk in PSI_KEYS:
            res = st.compare(d["V"], d["F"], d["P"], d[pk], d["corner_dist"], cones, theta, r, K_int=d["K_int"])
            fields[(lay, pk)] = (res.pop("_diff"), res.pop("_masked"), cones)
            rows.append({"id": sid, "group": group, "layout": lay, "psi_key": pk, "n_vertices": len(d["V"]),
                         "K_int_vs_angle_defect": k_err, "error": "", **res})
    return rows, fields


def _fmt(x):
    return "" if pd.isna(x) else f"{x:.3g}"


def write_markdown(df: pd.DataFrame, path: Path, args) -> None:
    ok = df[df["error"] == ""]
    lines = ["# Closed-form u vs Poisson solve", "",
             f"manifest `{args.manifest}`, processed `{args.processed}`, n = {df['id'].nunique()} shapes, "
             f"exclude radius {args.exclude_radius}, cone min separation {args.min_sep}, seed {args.seed}.", "",
             "Errors are |u_closed - u_poisson| after removing the area-weighted mean difference. `masked`: corner "
             "mask and cone disks excluded; `unmasked`: cone disks only. `u_range`: range of u (masked set).", "",
             "## Per source group (median / max over shapes)", "",
             "| group | layout | psi | n | p99 masked | max masked | p99 unmasked | max unmasked | u_range |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (g, lay, pk), s in ok.groupby(["group", "layout", "psi_key"], sort=False):
        cell = lambda c: f"{_fmt(s[c].median())} / {_fmt(s[c].max())}"  # noqa: E731
        lines.append(f"| {g} | {lay} | {pk} | {len(s)} | {cell('masked_p99')} | {cell('masked_max')} | "
                     f"{cell('unmasked_p99')} | {cell('unmasked_max')} | {cell('u_range')} |")
    err = df[df["error"] != ""]
    if len(err):
        lines += ["", "## Layouts not built", ""] + [f"- {r.id} {r.layout}: {r.error}" for r in err.itertuples()]
    lines += ["", "## Per shape", "", "| id | group | layout | psi | p99 m | max m | rms m | p99 u | max u | "
              "area kept | u_range |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok.itertuples():
        lines.append(f"| {r.id} | {r.group} | {r.layout} | {r.psi_key} | {_fmt(r.masked_p99)} | {_fmt(r.masked_max)} | "
                     f"{_fmt(r.masked_rms)} | {_fmt(r.unmasked_p99)} | {_fmt(r.unmasked_max)} | "
                     f"{_fmt(r.masked_area_share)} | {_fmt(r.u_range)} |")
    path.write_text("\n".join(lines) + "\n")


def plot_fields(examples, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples), len(LAYOUTS), figsize=(11, 3.2 * len(examples)), squeeze=False)
    for row, (sid, P, fields) in zip(axes, examples):
        for ax, lay in zip(row, LAYOUTS):
            if (lay, "psi") not in fields:
                ax.set_axis_off()
                continue
            diff, masked, cones = fields[(lay, "psi")]
            lat, lon = np.degrees(np.arcsin(np.clip(P[:, 2], -1, 1))), np.degrees(np.arctan2(P[:, 1], P[:, 0]))
            ax.scatter(lon[~masked], lat[~masked], s=1, c="0.8", rasterized=True)
            vmax = max(np.percentile(np.abs(diff[masked]), 99), 1e-6)
            sc = ax.scatter(lon[masked], lat[masked], s=1, c=diff[masked], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            rasterized=True)
            ax.scatter(lon[cones], lat[cones], marker="x", c="k", s=30)
            ax.set(title=f"{sid} {lay} (psi)", xlim=(-180, 180), ylim=(-90, 90), xlabel="lon", ylabel="lat")
            fig.colorbar(sc, ax=ax, label="u_closed - u_poisson")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/reports/pilot/manifest.parquet")
    ap.add_argument("--processed", default="data/processed/pilot")
    ap.add_argument("--out", default="data/reports/pilot_stretch")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--exclude-radius", type=float, default=0.025)
    ap.add_argument("--min-sep", type=float, default=0.2, help="layout b: min 3D distance between cones")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picks = pick_shapes(pd.read_parquet(args.manifest), args.n, args.seed)
    rows, examples = [], {}
    for sid, group in picks.itertuples(index=False):
        d = read_processed(Path(args.processed) / f"{sid}.npz")
        if "psi_lsq" not in d or "corner_dist" not in d:
            raise KeyError(f"{sid}: processed file lacks psi_lsq/corner_dist")
        r, fields = run_shape(sid, group, d, args.exclude_radius, args.min_sep)
        rows += r
        examples[sid] = (sid, d["P"], fields)
        print(f"{sid:28s} {group:16s} " + "  ".join(
            f"{x['layout']}/{x['psi_key']}: p99 {x['masked_p99']:.3g}" for x in r if not x["error"]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "stretch.csv", index=False)
    write_markdown(df, out / "STRETCH.md", args)
    ok = df[(df["error"] == "") & (df["psi_key"] == "psi") & (df["layout"] == "a_fps8")]
    ranked = ok.sort_values("masked_p99")["id"]
    worst, typical = ranked.iloc[-1], ranked.iloc[len(ranked) // 2]          # worst and median shape (layout a)
    plot_fields([examples[typical], examples[worst]] if typical != worst else [examples[worst]],
                out / "stretch_error_field.png")
    print(f"wrote {out}/stretch.csv, STRETCH.md, stretch_error_field.png")


if __name__ == "__main__":
    main()
