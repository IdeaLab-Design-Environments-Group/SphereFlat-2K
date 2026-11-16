"""grid stage: lat-long channels (C, H, W) of the sphere map in the stored (identity) frame.

Channels and their definitions are in ``moe_data.grid.compute_grid``; the order
comes from ``grid.channels`` and is stored in ``grid_channels``. The psi channels use the per-vertex
field named by ``grid.psi_source``. Rejects (accept-if form) when any cell centre is not contained in a
face (the locator needs every det[p0,p1,p2] > 0, the validate_sphere gate) or a channel is non-finite.
``mass_error`` = |sum K_int_density * Omega - 4 pi| is gated at ``grid.mass_tol``: each vertex's kernel is
normalised by its own discrete quadrature sum, so the total is exact up to rounding.
"""
import numpy as np

from moe_data import grid as gr
from moe_data.stages.base import StageResult

NAME = "grid"


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    gc = cfg["grid"]
    psi_key = gc["psi_source"]
    missing = [k for k in ("P", "F", "K_int", "K_pt", "H", psi_key) if inputs.get(k) is None]
    if missing:
        return StageResult(False, reason=f"missing_input: {', '.join(missing)}")
    channels = tuple(gc["channels"])
    G, info = gr.compute_grid(
        np.asarray(inputs["P"], np.float64), np.asarray(inputs["F"], np.int64),
        np.asarray(inputs[psi_key], np.float64), np.asarray(inputs["K_pt"], np.float64),
        np.asarray(inputs["H"], np.float64), np.asarray(inputs["K_int"], np.float64),
        int(gc["height"]), int(gc["width"]), float(gc["splat_alpha_cells"]), channels)
    meta = {"mass_error": info["mass_error"], "n_unlocated": info["n_unlocated"],
            "n_locator_fallback": info["n_fallback"]}
    finite = np.isfinite(G)
    if finite.all():
        for c, name in enumerate(channels):
            meta[f"{name}_min"] = float(G[c].min())
            meta[f"{name}_max"] = float(G[c].max())
    fails = []
    if not info["n_unlocated"] == 0:
        fails.append(f"locate_failed: {info['n_unlocated']} cell centres in no face")
    if not (np.isfinite(info["mass_error"]) and info["mass_error"] <= cfg["grid"]["mass_tol"]):
        fails.append(f"grid_mass: |sum density*Omega - 4pi| = {info['mass_error']:.3g}")
    n_bad = int(np.sum(~finite))
    if not n_bad == 0:
        bad = [channels[c] for c in range(len(channels)) if not finite[c].all()]
        fails.append(f"nonfinite_grid: {n_bad} values in {', '.join(bad)}")
    if fails:
        return StageResult(False, meta=meta, reason="; ".join(fails))
    return StageResult(True, data={"grid": G.astype(np.float32), "grid_channels": np.array(channels, dtype=str)},
                       meta=meta)
