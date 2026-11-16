"""Load stage for procedural shapes: generate the mesh from ``inputs["candidate"]``."""
import json

from moe_data.procedural import generate
from moe_data.stages.base import StageResult

NAME = "load"


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    cand = inputs.get("candidate")
    if not isinstance(cand, dict) or "family" not in cand:
        return StageResult(False, reason="missing_input: no procedural candidate dict")
    V, F, params = generate(cand["family"], int(cand["index"]), cfg)
    meta = {"family": cand["family"], "index": int(cand["index"]),
            "params_json": json.dumps(params, sort_keys=True),
            "n_vertices_raw": int(V.shape[0]), "n_faces_raw": int(F.shape[0])}
    for k in ("max_length", "max_length_over_radius"):
        if k in params:
            meta[k] = params[k]
    return StageResult(True, data={"V": V, "F": F}, meta=meta)
