"""Package the processed dataset for a public release.

Writes to --out: manifest.parquet (every candidate except licence-excluded ones, with Thingi10K file and
thing ids, author and licence for attribution), the accepted shapes as uncompressed tar shards of the
per-shape .npz files (each shard below --max-shard-gb), and SHA256SUMS.

Usage: uv run python scripts/package_release.py --out data/release/v1.0
"""
import argparse
import hashlib
import tarfile
from pathlib import Path

import pandas as pd

from moe_data.config import load_config

DROP_COLUMNS = ("traceback", "original_file", "load.file_path", "key")


def public_manifest(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["reject_stage"] != "licence"].drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    leaked = [c for c in out.columns
              if out[c].astype(str).str.contains(r"/Users/|/home/|\\Users\\", regex=True).any()]
    if leaked:
        raise ValueError(f"local paths in columns {leaked}")
    return out.reset_index(drop=True)


def shard(files: list[Path], max_bytes: int) -> list[list[Path]]:
    shards, cur, size = [], [], 0
    for f in files:
        n = f.stat().st_size
        if cur and size + n > max_bytes:
            shards.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += n
    return shards + ([cur] if cur else [])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="SphereFlat-2K")
    ap.add_argument("--max-shard-gb", type=float, default=1.9)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = public_manifest(pd.read_parquet(cfg["paths"]["manifest"]))
    accepted = df[df["ok"].astype(bool)]
    processed = Path(cfg["paths"]["processed"]) / "full"
    files = [processed / f"{sid}.npz" for sid in sorted(accepted["id"])]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} accepted shapes have no processed file, e.g. {missing[0]}")
    df.to_parquet(out / "manifest.parquet", index=False)
    written = [out / "manifest.parquet"]
    for k, group in enumerate(shard(files, int(args.max_shard_gb * 1e9)), 1):
        path = out / f"{args.name}_part{k}.tar"
        with tarfile.open(path, "w") as tar:
            for f in group:
                tar.add(f, arcname=f"{args.name}/{f.name}")
        written.append(path)
        print(f"{path.name}: {len(group)} shapes, {path.stat().st_size / 1e9:.2f} GB")
    (out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in written))
    print(f"{len(accepted)} shapes, {len(df)} manifest rows -> {out}")


if __name__ == "__main__":
    main()
