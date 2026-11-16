"""Topology stage: keep only closed, manifold, oriented, single-component genus-0 meshes.

Passes on the cleaned (unreferenced vertices removed) and outward-oriented mesh.
"""
from moe_data.stages.base import StageResult
from moe_data.topology import check_and_orient

NAME = "topology"


def run(shape_id: str, inputs: dict, cfg: dict) -> StageResult:
    V, F = inputs.get("V"), inputs.get("F")
    if V is None or F is None:
        return StageResult(False, reason="missing_input: V or F not in inputs")
    ok, reason, measures, V2, F2 = check_and_orient(V, F)
    if not ok:
        return StageResult(False, meta=measures, reason=reason)
    return StageResult(True, data={"V": V2, "F": F2}, meta=measures)
