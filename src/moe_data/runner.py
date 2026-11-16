"""Resumable per-shape stage runner.

Each stage output is stored as ``work/<stage>/<id>.json`` (status, meta, reason, cache key) plus
``work/<stage>/<id>.npz`` (data arrays) when the stage succeeded. A stored result is reused only when its
cache key matches. The key hashes the stage's source files, the config sections it reads and the
upstream stage's key, so changing a threshold, the code or an earlier stage recomputes this stage and every
later one. Rejections are cached the same way, so a relaxed threshold re-admits shapes.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import resource
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StageSpec:
    module: str                 # e.g. "moe_data.stages.topology"
    cfg_keys: tuple[str, ...]   # top-level config sections the stage reads
    deps: tuple[str, ...] = ()  # extra source files (relative to the package) the stage depends on
    store_data: bool = True     # False: cache only the record; arrays are regenerated when a later stage needs them
    external: tuple[str, ...] = ()  # keys of cfg["paths"] naming external files (e.g. a binary) hashed into the key

    @property
    def name(self) -> str:
        return importlib.import_module(self.module).NAME


def _file_digest(rel: str) -> str:
    return hashlib.sha256((PKG_DIR / rel).read_bytes()).hexdigest()


_EXT_CACHE: dict = {}


def _external_digest(path: str) -> str:
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _EXT_CACHE:
        _EXT_CACHE[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    return _EXT_CACHE[key]


def _library_versions() -> str:
    from importlib.metadata import PackageNotFoundError, version
    out = []
    for pkg in ("numpy", "scipy", "trimesh", "pymeshlab", "libigl", "torch"):
        try:
            out.append(f"{pkg}={version(pkg)}")
        except PackageNotFoundError:
            out.append(f"{pkg}=none")
    return ",".join(out)


def root_key(candidate: dict) -> str:
    """Key of a shape's raw input: candidate metadata, the raw file's identity (path, size, mtime) and the
    numerical library versions, so a changed file, metadata or library recomputes the whole chain."""
    parts = [json.dumps(candidate, sort_keys=True, default=str), _library_versions()]
    fp = candidate.get("file_path")
    if fp:
        st = Path(fp).stat()
        parts.append(f"{Path(fp).resolve()}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


def stage_key(spec: StageSpec, cfg: dict, upstream_key: str, extra: str = "") -> str:
    mod_rel = spec.module.removeprefix("moe_data.").replace(".", "/") + ".py"
    parts = [spec.module, _file_digest(mod_rel)]
    parts += [_file_digest(d) for d in spec.deps]
    parts += [_external_digest(cfg["paths"][k]) for k in spec.external]
    parts.append(json.dumps({k: cfg.get(k) for k in spec.cfg_keys}, sort_keys=True, default=str))
    parts += [upstream_key, extra]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


def _paths(work: Path, stage: str, shape_id: str) -> tuple[Path, Path]:
    d = work / stage
    return d / f"{shape_id}.json", d / f"{shape_id}.npz"


def _load_cached(work: Path, stage: str, shape_id: str, key: str):
    jp, npzp = _paths(work, stage, shape_id)
    if not jp.exists():
        return None
    rec = json.loads(jp.read_text())
    if rec.get("key") != key:
        return None
    data = {}
    if rec["ok"] and rec.get("data_stored", True):
        if not npzp.exists():
            return None
        with np.load(npzp, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
    return rec, data


def _store(work: Path, stage: str, shape_id: str, rec: dict, data: dict, store_data: bool = True) -> None:
    jp, npzp = _paths(work, stage, shape_id)
    jp.parent.mkdir(parents=True, exist_ok=True)
    rec["data_stored"] = store_data
    if rec["ok"] and store_data:  # always written on success, even when empty, so the cache lookup finds it
        tmp = npzp.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.replace(npzp)
    elif npzp.exists():
        npzp.unlink()
    tmp = jp.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(rec, default=_json_default))
    tmp.replace(jp)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def run_chain(shape_id: str, seed_inputs: dict, specs: list[StageSpec], cfg: dict,
              work: Path, upstream_key: str = "root") -> dict:
    """Run ``specs`` in order for one shape. Returns a manifest fragment.

    The fragment holds ``ok``, ``reject_stage``, ``reject_reason``, ``key`` (last stage key),
    per-stage meta under ``<stage>.<field>`` and timings under ``time.<stage>``.
    """
    inputs = dict(seed_inputs)
    row: dict = {"id": shape_id, "ok": True, "reject_stage": "", "reject_reason": ""}
    key = upstream_key
    history: list = []          # (module, data or None) per completed stage; None = arrays not stored
    stale = False               # True while inputs lack the arrays of a cached store_data=False stage
    for spec in specs:
        mod = importlib.import_module(spec.module)
        name = mod.NAME
        key = stage_key(spec, cfg, key)
        cached = _load_cached(work, name, shape_id, key)
        if cached is None:
            if stale:                                # rebuild inputs in pipeline order: replay uncached
                inputs = dict(seed_inputs)           # stages, then reapply later cached outputs on top
                for hmod, hdata in history:
                    inputs.update(hmod.run(shape_id, inputs, cfg).data if hdata is None else hdata)
                stale = False
            t0 = time.perf_counter()
            try:
                res = mod.run(shape_id, inputs, cfg)
                rec = {"ok": bool(res.ok), "meta": res.meta, "reason": res.reason, "key": key}
                data = res.data
            except Exception as e:  # a crash rejects this shape only
                rec = {"ok": False, "meta": {}, "key": key,
                       "reason": f"crash: {type(e).__name__}: {e}",
                       "traceback": traceback.format_exc(limit=8)}
                data = {}
            rec["seconds"] = time.perf_counter() - t0
            rec["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
            if not rec["reason"].startswith("crash:"):     # crashes (OOM, transient) are retried next run
                _store(work, name, shape_id, rec, data, spec.store_data)
        else:
            rec, data = cached
            if rec["ok"] and not rec.get("data_stored", True):
                stale = True
                data = None
        row[f"time.{name}"] = rec.get("seconds", np.nan)
        for k, v in (rec.get("meta") or {}).items():
            if np.isscalar(v) or v is None:
                row[f"{name}.{k}"] = v
        if not rec["ok"]:
            row.update(ok=False, reject_stage=name, reject_reason=rec["reason"])
            if "traceback" in rec:
                row["traceback"] = rec["traceback"]
            break
        history.append((mod, data))
        if data is not None:
            inputs.update(data)
    row["key"] = key
    return row


def load_stage_data(work: Path, stage: str, shape_id: str) -> dict:
    _, npzp = _paths(work, stage, shape_id)
    with np.load(npzp, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
