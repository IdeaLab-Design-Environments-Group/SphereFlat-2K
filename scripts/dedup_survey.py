"""Measurement only: phase A (load, topology, suitability, normalize) + shape descriptors on ALL Thingi10K
files, to choose the near-duplicate threshold from the nearest-neighbour distance distribution.

Writes data/reports/survey/phase_a.parquet and descriptors.npz. No remeshing, no sphere maps.
Usage: uv run python scripts/dedup_survey.py --config config.yaml
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from moe_data import thingi
from moe_data.config import load_config
from moe_data.run import PHASE_A_THINGI, _phase_a, _pool_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    work = Path(cfg["paths"]["work"]) / "full"            # same cache the full run will reuse
    out = Path(cfg["paths"]["reports"]) / "survey"
    out.mkdir(parents=True, exist_ok=True)
    cands = thingi.list_candidates(cfg, pilot=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    jobs = [(c["shape_id"], {**c, "source": "thingi10k"}, PHASE_A_THINGI, cfg, str(work)) for c in cands]
    rows = _pool_map(_phase_a, jobs, cfg, "survey")
    ok = [r for r in rows if r.get("ok") and "_descriptor" in r]
    np.savez_compressed(out / "descriptors.npz", ids=np.array([r["id"] for r in ok]),
                        D=np.array([r["_descriptor"] for r in ok]),
                        hashes=np.array([r["_hash"] for r in ok]),
                        thing=np.array([r["thing_id"] for r in ok]),
                        n_faces=np.array([r["_n_faces"] for r in ok]))
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df.to_parquet(out / "phase_a.parquet", index=False)
    print(df["ok"].sum(), "of", len(df), "pass phase A;",
          df.loc[~df["ok"], "reject_stage"].value_counts().to_dict())


if __name__ == "__main__":
    main()
