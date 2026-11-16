"""moe_data.report: helpers, and a smoke run on a tiny synthetic manifest + processed files."""
import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import trimesh
import yaml

from moe_data import report
from moe_data.dataset import write_processed

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
BASE_CFG = yaml.safe_load(CONFIG.read_text())
CHANNELS = ["psi", "K_pt"]
# values the assertions below depend on, set here so the tests do not follow edits to config.yaml
TEST_OVERRIDES = {("dedup", "near_threshold"): 0.02, ("split", "holdout_by"): "length_over_radius",
                  ("split", "holdout_length_min"): 2.4, ("split", "holdout_lr_min"): 16,
                  ("protrusion", "range_threshold"): 6.0, ("protrusion", "range_source"): "psi_nocorner_range",
                  ("grid", "mass_tol"): 1e-8}


def _test_cfg(**extra) -> dict:
    cfg = copy.deepcopy(BASE_CFG)
    for (section, key), v in {**TEST_OVERRIDES, **extra}.items():
        cfg.setdefault(section, {})[key] = v
    return cfg


def test_reason_code_strips_numbers_and_joins_failures():
    assert report.reason_code("too_few_faces: 124 < 500") == "too_few_faces"
    assert report.reason_code("flipped_triangles: 3 x; not_centered: |r| = 1e-3") == \
        "flipped_triangles + not_centered"
    assert report.reason_code("crash: ValueError: bad") == "crash"
    assert report.reason_code("") == "(empty)"


def test_stage_flow_counts_survivors_in_order():
    df = pd.DataFrame({"id": list("abcde"), "ok": [False, False, False, True, True],
                       "reject_stage": ["topology", "topology", "remesh", "", ""]})
    flow = {r["stage"]: r for r in report.stage_flow(df, list(report.STAGE_ORDER))}
    assert flow["load"] == {"stage": "load", "in": 5, "rejected": 0, "out": 5}
    assert flow["topology"]["out"] == 3
    assert flow["remesh"] == {"stage": "remesh", "in": 3, "rejected": 1, "out": 2}
    assert flow["visibility"]["out"] == 2


def test_stage_flow_keeps_unknown_stages():
    df = pd.DataFrame({"id": ["a", "b"], "ok": [False, False], "reject_stage": ["worker", None]})
    stages = report.stage_list(df)
    assert stages[-2:] == ["unknown", "worker"]
    flow = report.stage_flow(df, stages)
    assert flow[-1]["out"] == 0


def test_parse_proposed_marks_sections_and_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\nsuitability:   # PROPOSED\n  min_faces: 500\nremesh:\n  target_vertices: 15000\n"
                 "  hausdorff_tol: 0.01   # PROPOSED, relative\n  note: 1  # not proposed\n")
    prop = set(report.parse_proposed(p))
    assert prop == {("suitability", "*"), ("remesh", "hausdorff_tol")}


def test_threshold_pairs_picks_nearest_on_each_side():
    pairs = [(0, 1, 0.001), (2, 3, 0.0049), (4, 5, 0.0045), (6, 7, 0.005), (8, 9, 0.0051), (10, 11, 0.02)]
    below, above = report.threshold_pairs(pairs, 0.005, 2)
    assert [p[2] for p in below] == [0.0049, 0.0045]          # highest below first
    assert [p[2] for p in above] == [0.005, 0.0051]           # the threshold itself counts as above
    below, above = report.threshold_pairs(pairs, 1.0, 3)
    assert len(below) == 3 and above == []
    assert report.threshold_pairs([], 0.005, 6) == ([], [])


def test_folded_mask_uses_flag_or_reason():
    df = pd.DataFrame({"reject_stage": ["validate_sphere", "validate_sphere", "remesh", ""],
                       "reject_reason": ["flipped_triangles: 3 x", "degenerate_sphere_triangles: 2",
                                         "flipped_triangles", ""]})
    assert report.folded_mask(df).tolist() == [True, False, False, False]
    df["sphere_folded"] = [False, True, False, None]
    assert report.folded_mask(df).tolist() == [False, True, False, False]


def test_is_final_run(tmp_path):
    cfg = {"paths": {"manifest": str(tmp_path / "manifest.parquet")}}
    assert report.is_final_run(cfg, tmp_path / "manifest.parquet", tmp_path / "processed" / "pilot")
    assert report.is_final_run(cfg, tmp_path / "other.parquet", tmp_path / "processed" / "full")
    assert not report.is_final_run(cfg, tmp_path / "reports" / "pilot" / "m.parquet", tmp_path / "processed" / "pilot")


def test_quantile_cells_ignores_nonfinite_and_handles_empty():
    cells = report.quantile_cells(pd.Series([1.0, np.nan, np.inf, 3.0]))
    assert cells[0] == 2 and cells[1] == "1" and cells[-1] == "3"
    assert report.quantile_cells(pd.Series([], dtype=float)) == [0] + ["not available"] * 5


def _sphere_file(path: Path, scale: float, gh: int = 8, gw: int = 16) -> None:
    m = trimesh.creation.icosphere(subdivisions=2)
    P = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    F = np.asarray(m.faces, np.int64)
    V = P * np.array([scale, 1.0, 1.0])
    n = len(P)
    psi = np.log(np.linspace(0.5, 1.5, n))
    grid = np.stack([np.linspace(-1, 1, gh * gw).reshape(gh, gw)] * len(CHANNELS)).astype(np.float32)
    write_processed(path, {"V": V, "F": F, "P": P, "area": np.full(n, 1.0 / n), "area_sphere": np.full(n, 1.0 / n),
                           "psi": psi, "psi_lsq": psi, "K_int": np.zeros(n), "K_pt": np.zeros(n), "H": np.ones(n),
                           "vis": np.ones((n, 2), np.float32), "grid": grid, "grid_channels": CHANNELS})


def _manifest() -> pd.DataFrame:
    prot = json.dumps({"protrusions": [{"length": 2.5, "thickness": 0.1}]})
    rows = [
        {"id": "thingi_1", "ok": False, "reject_stage": "topology", "reject_reason": "not_genus0: chi=0 genus=1",
         "source": "thingi10k", "time.load": 0.01, "time.topology": 0.02},
        {"id": "thingi_2", "ok": False, "reject_stage": "suitability", "reject_reason": "too_few_faces: 10 < 500",
         "source": "thingi10k", "time.load": 0.01, "suitability.n_faces": 10.0},
        {"id": "proc_blob_0", "ok": True, "reject_stage": "", "reject_reason": "", "source": "procedural",
         "load.family": "blob", "split": "train", "time.load": 0.1, "time.spheremap": 2.0,
         "psi.psi_range": 0.2, "validate_sphere.centering_residual": 1e-12, "suitability.n_faces": 30000.0},
        {"id": "proc_prot_0", "ok": True, "reject_stage": "", "reject_reason": "", "source": "procedural",
         "load.family": "protrusion", "load.params_json": prot, "split": "protrusion_test", "time.load": 3.0,
         "time.spheremap": 2.5, "psi.psi_range": 8.0, "validate_sphere.centering_residual": 2e-12,
         "suitability.n_faces": 30000.0, "psi.psi_nocorner_range": 7.5, "psi.corner_area_share": 0.01,
         "psi.psi_vs_lsq_nocorner_max_abs": 0.05, "grid.mass_error": 3e-12},
        {"id": "thingi_3", "ok": False, "reject_stage": "validate_sphere",
         "reject_reason": "flipped_triangles: 5 with negative orientation", "source": "thingi10k",
         "validate_sphere.n_flipped": 5.0, "validate_sphere.min_orient": -1e-4, "time.load": 0.01},
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def run_dir(tmp_path, monkeypatch):
    cfg = _test_cfg()
    cfg["paths"]["work"] = str(tmp_path / "work")
    cfg["paths"]["reports"] = str(tmp_path / "reports")
    cfg["paths"]["manifest"] = str(tmp_path / "final_manifest.parquet")
    cfg_path = tmp_path / "cfg_used.yaml"
    # one PROPOSED marker, so the decisions section has a threshold to list
    text = yaml.safe_dump(cfg)
    cfg_path.write_text(re.sub(r"(?m)^(  hausdorff_tol: .*)$", r"\1  # PROPOSED", text))
    proc = tmp_path / "processed"
    _sphere_file(proc / "proc_blob_0.npz", 1.0)
    _sphere_file(proc / "proc_prot_0.npz", 2.0)
    man = tmp_path / "manifest.parquet"
    _manifest().to_parquet(man, index=False)
    from moe_data import thingi

    def no_metadata(*a, **k):
        raise FileNotFoundError("no Thingi10K metadata in tests")
    monkeypatch.setattr(thingi, "list_candidates", no_metadata)
    return tmp_path, cfg_path, man, proc


def test_report_smoke_without_survey(run_dir):
    tmp, cfg_path, man, proc = run_dir
    out = tmp / "out"
    report.main(["--config", str(cfg_path), "--manifest", str(man), "--processed", str(proc), "--out", str(out)])
    text = (out / "REPORT.md").read_text()
    assert text.index("## Decisions needed") < text.index("## 1. Counts")
    for h in ("## 2. Threshold", "## 3. Timing", "## 4. Shapes", "## 5. Validation", "## 6. Splits",
              "## 7. psi", "## 8. Curvature", "## 9. psi vs psi_lsq", "## 10. Grid mass", "## 11. Disk use",
              "## 12. Stretch check"):
        assert h in text
    assert text.startswith("# Dataset pipeline report")             # not the full run
    assert "## Folded sphere maps" not in text
    assert "| protrusion_test_inrange | 0 | 0 | 0 | 0 | 0 |" in text  # label listed though this run has none
    assert "| protrusion_test | 0 | 0 | 1 | 1 |" in text
    assert "| psi range outside corners | proc_protrusion | 1 |" in text
    assert "| max \\|K_int\\| | proc_blob | 1 | 0 |" in text          # read from the processed file
    assert "proc_protrusion: 0 of 1 candidates rejected." in text     # per-group rejection tables
    assert "| proc_protrusion | 1 | 1 |" in text                      # disk use per group
    assert "not available" in text                  # missing columns (hausdorff, timing of most stages, survey)
    assert "| topology | not_genus0 | 1 |" in text
    assert "thingi10k / validate_sphere: 1 of 1 rejected" in text  # > 50% flag
    assert "thingi10k / suitability" not in text                   # exactly 50% is not flagged
    assert "thingi10k / topology" not in text
    assert "| centering residual \\|r\\| | max 2e-12 |" in text
    for name in ("suitability.png", "shapes_procedural.png", "psi_protrusion.png"):
        assert (out / "fig" / name).stat().st_size > 0
    assert "of 1 protrusion shapes, 1 held out by length >= 2.4, 1 by L/r > 16" in text


def test_report_with_survey(run_dir):
    tmp, cfg_path, man, proc = run_dir
    survey = tmp / "survey"
    survey.mkdir()
    ids = np.array(["thingi_10", "thingi_11", "thingi_12", "thingi_13"])
    D = np.array([[0.0, 0.0], [0.001, 0.0], [0.5, 0.0], [0.5, 0.004]])
    np.savez_compressed(survey / "descriptors.npz", ids=ids, D=D, hashes=np.array(["a", "b", "c", "d"]),
                        thing=np.array([1, 2, 3, 3]), n_faces=np.array([600, 700, 800, 900]))
    pd.DataFrame({"id": [*ids, "thingi_14"], "ok": [True] * 4 + [False],
                  "reject_stage": [""] * 4 + ["topology"], "reject_reason": [""] * 4 + ["not_closed: 3"],
                  "source": "thingi10k", "time.load": 0.01, "time.topology": 0.02,
                  "suitability.n_faces": [600.0, 700.0, 800.0, 900.0, np.nan]}).to_parquet(survey / "phase_a.parquet")
    meshes = tmp / "meshes"
    m = trimesh.creation.icosphere(subdivisions=1)
    for i in ids[:2]:
        meshes.mkdir(exist_ok=True)
        np.savez(meshes / f"{i}.npz", V=np.asarray(m.vertices), F=np.asarray(m.faces, np.int64))
    out = tmp / "out"
    report.main(["--config", str(cfg_path), "--manifest", str(man), "--processed", str(proc), "--out", str(out),
                 "--survey", str(survey), "--survey-meshes", str(meshes)])
    text = (out / "REPORT.md").read_text()
    # cross-thing NN: 10<->11 at 0.001; 12-13 share thing 3, so their cross-thing NN is ~0.5 away,
    # but they are 0.004 apart and dedup merges them as near duplicates: 2 of 4 kept
    assert "| 0.002 | 2 (50.0%) | 1 | 2 | 2 |" in text
    assert "dedup_records on the survey at near_threshold 0.02: 2 of 4 kept" in text
    assert "Projected wall time:" in text
    assert (out / "fig" / "dedup_pairs.png").exists()
    # pairs around near_threshold 0.02: below = 10-11 (0.001); above = 11-12 and 11-13 (~0.499)
    assert "| below (merged) | 1 | `thingi_10` | 1 | `thingi_11` | 2 | 0.00100 | 600 / 700 |" in text
    assert "| above (kept apart) | 1 | `thingi_11` | 2 | `thingi_12` | 3 | 0.49900 |" in text
    assert "| above (kept apart) | 3 |" not in text
    assert (out / "fig" / "dedup_threshold_pairs.png").stat().st_size > 0
    assert (out / "dedup_threshold_pairs.png").stat().st_size > 0      # standalone copy
    assert "meshes missing" in text                                  # thingi_12/13 have no mesh here


def test_final_report_layout(run_dir):
    tmp, cfg_path, man, proc = run_dir
    final_man = tmp / "final_manifest.parquet"                       # = paths.manifest in the test config
    df = pd.read_parquet(man)
    df["sphere_folded"] = df["id"] == "thingi_3"
    df.to_parquet(final_man, index=False)
    stretch = tmp / "reports" / "processed" / "stretch" / "STRETCH.md"
    stretch.parent.mkdir(parents=True)
    stretch.write_text("| shape | max stretch |\n|---|---|\n| proc_blob_0 | 1.7 |\n")
    out = tmp / "final"
    report.main(["--config", str(cfg_path), "--manifest", str(final_man), "--processed", str(proc),
                 "--out", str(out)])
    text = (out / "REPORT.md").read_text()
    assert text.startswith("# Final dataset report")
    order = ["## Final dataset", "## Folded sphere maps", "## Rejection reasons", "## Items to inspect by hand",
             "## Decisions needed", "## 1. Counts"]
    assert [text.index(h) for h in order] == sorted(text.index(h) for h in order)
    assert "| thingi10k | 1 | 1 | 100.0% |" in text                    # folded count / reached validate
    assert "| `thingi_3` | thingi10k | 5 | -0.0001 |" in text
    decisions = text[text.index("## Decisions needed"):text.index("## 1. Counts")]
    assert "PROPOSED `remesh.hausdorff_tol`" in decisions
    assert "grid.psi_source" not in decisions and "Approve the full run" not in decisions
    assert "### Projected full run" not in text
    assert "Largest corner area share among accepted" in text and "`proc_prot_0` | proc_protrusion | 0.01" in text
    assert "## 12. Stretch check" in text and "| proc_blob_0 | 1.7 |" in text      # included verbatim


def test_final_mode_can_be_forced_off(run_dir):
    tmp, cfg_path, man, proc = run_dir
    out = tmp / "out"
    report.main(["--config", str(cfg_path), "--manifest", str(man), "--processed", str(proc), "--out", str(out),
                 "--final", "no", "--stretch", str(tmp / "missing.md")])
    text = (out / "REPORT.md").read_text()
    assert text.startswith("# Dataset pipeline report")
    assert "not available (no stretch summary at" in text
