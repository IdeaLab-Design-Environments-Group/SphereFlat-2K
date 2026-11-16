"""Load stage: read a Thingi10K mesh from the candidate dict in ``inputs["candidate"]``."""
from moe_data.stages.base import StageResult
from moe_data.thingi import load_mesh

NAME = "load"

# small JSON-able candidate fields copied into the manifest row
_META_KEYS = ("file_id", "thing_id", "licence", "redistributable", "author",
              "file_path", "self_intersecting")


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    cand = inputs.get("candidate")
    if not isinstance(cand, dict) or "file_path" not in cand:
        return StageResult(False, reason="missing_input: no candidate dict with file_path")
    meta = {k: cand[k] for k in _META_KEYS if k in cand}
    try:
        V, F = load_mesh(cand)
    except (OSError, ValueError) as e:
        return StageResult(False, meta=meta, reason=f"load_failed: {type(e).__name__}: {e}")
    meta.update(n_vertices_raw=int(V.shape[0]), n_faces_raw=int(F.shape[0]))
    return StageResult(True, data={"V": V, "F": F}, meta=meta)
