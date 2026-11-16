import json
import sys
import textwrap

import numpy as np

from moe_data import runner


def _make_stage(tmp_path, name, body):
    pkg = tmp_path / "fakestages"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / f"{name}.py").write_text(textwrap.dedent(body))
    return f"fakestages.{name}"


def _setup(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(runner, "PKG_DIR", tmp_path)
    for m in [m for m in sys.modules if m.startswith("fakestages")]:
        del sys.modules[m]
    counter = tmp_path / "calls.txt"
    double = _make_stage(tmp_path, "double", f"""
        from moe_data.stages.base import StageResult
        NAME = "double"
        def run(shape_id, inputs, cfg):
            open({str(counter)!r}, "a").write("double\\n")
            return StageResult(True, data={{"x": inputs["x"] * 2}}, meta={{"n": 1}})
        """)
    gate = _make_stage(tmp_path, "gate", f"""
        from moe_data.stages.base import StageResult
        NAME = "gate"
        def run(shape_id, inputs, cfg):
            open({str(counter)!r}, "a").write("gate\\n")
            ok = bool(inputs["x"].sum() <= cfg["gate"]["max"])
            return StageResult(ok, meta={{"s": float(inputs["x"].sum())}}, reason="" if ok else "too_big: x")
        """)
    specs = [runner.StageSpec(double, ()), runner.StageSpec(gate, ("gate",))]
    return specs, counter


def _calls(counter):
    return counter.read_text().split() if counter.exists() else []


def test_resume_reuses_cached_results(tmp_path, monkeypatch):
    specs, counter = _setup(tmp_path, monkeypatch)
    cfg = {"gate": {"max": 100}}
    seed = {"x": np.arange(3.0)}
    r1 = runner.run_chain("s1", seed, specs, cfg, tmp_path / "work")
    r2 = runner.run_chain("s1", seed, specs, cfg, tmp_path / "work")
    assert r1["ok"] and r2["ok"]
    assert _calls(counter) == ["double", "gate"]          # second run fully cached
    assert r1["key"] == r2["key"]


def test_config_change_invalidates_cached_rejection(tmp_path, monkeypatch):
    specs, counter = _setup(tmp_path, monkeypatch)
    seed = {"x": np.arange(3.0)}                          # doubled sum = 6
    r1 = runner.run_chain("s1", seed, specs, {"gate": {"max": 1}}, tmp_path / "work")
    assert not r1["ok"] and r1["reject_stage"] == "gate" and r1["reject_reason"].startswith("too_big")
    r2 = runner.run_chain("s1", seed, specs, {"gate": {"max": 100}}, tmp_path / "work")
    assert r2["ok"]                                       # relaxed threshold re-admits the shape
    assert _calls(counter) == ["double", "gate", "gate"]  # upstream stage still cached


def test_code_change_invalidates_stage_and_descendants(tmp_path, monkeypatch):
    specs, counter = _setup(tmp_path, monkeypatch)
    cfg = {"gate": {"max": 100}}
    seed = {"x": np.arange(3.0)}
    runner.run_chain("s1", seed, specs, cfg, tmp_path / "work")
    src = tmp_path / "fakestages" / "double.py"
    src.write_text(src.read_text() + "\n# edited\n")
    runner.run_chain("s1", seed, specs, cfg, tmp_path / "work")
    assert _calls(counter) == ["double", "gate", "double", "gate"]


def test_crash_is_recorded_as_rejection(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    boom = _make_stage(tmp_path, "boom", """
        NAME = "boom"
        def run(shape_id, inputs, cfg):
            raise ValueError("bad input")
        """)
    row = runner.run_chain("s1", {}, [runner.StageSpec(boom, ())], {}, tmp_path / "work")
    assert not row["ok"]
    assert row["reject_stage"] == "boom"
    assert row["reject_reason"] == "crash: ValueError: bad input"
    assert "ValueError: bad input" in row["traceback"]
    assert not (tmp_path / "work" / "boom" / "s1.json").exists()      # crashes are not cached


def test_stage_without_stored_data_is_rerun_only_when_needed(tmp_path, monkeypatch):
    specs, counter = _setup(tmp_path, monkeypatch)
    specs[0] = runner.StageSpec(specs[0].module, specs[0].cfg_keys, store_data=False)   # "double"
    seed = {"x": np.arange(3.0)}
    r1 = runner.run_chain("s1", seed, specs, {"gate": {"max": 100}}, tmp_path / "work")
    assert r1["ok"]
    assert not (tmp_path / "work" / "double" / "s1.npz").exists()      # arrays not cached
    runner.run_chain("s1", seed, specs, {"gate": {"max": 100}}, tmp_path / "work")
    assert _calls(counter) == ["double", "gate"]                        # everything cached: nothing reruns
    r3 = runner.run_chain("s1", seed, specs, {"gate": {"max": 1}}, tmp_path / "work")
    assert not r3["ok"] and r3["gate.s"] == 6.0                         # gate saw the regenerated x
    assert _calls(counter) == ["double", "gate", "double", "gate"]


def test_replayed_uncached_stage_does_not_clobber_later_cached_outputs(tmp_path, monkeypatch):
    """Load (store_data=False) is replayed on demand; the cached outputs of the stages
    after it (topology-cleaned V,F) must win over the replayed raw arrays."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(runner, "PKG_DIR", tmp_path)
    for m in [m for m in sys.modules if m.startswith("fakestages")]:
        del sys.modules[m]
    raw = _make_stage(tmp_path, "raw", """
        import numpy as np
        from moe_data.stages.base import StageResult
        NAME = "raw"
        def run(shape_id, inputs, cfg):
            return StageResult(True, data={"x": np.array([-1.0, -2.0])})
        """)
    clean = _make_stage(tmp_path, "clean", """
        import numpy as np
        from moe_data.stages.base import StageResult
        NAME = "clean"
        def run(shape_id, inputs, cfg):
            return StageResult(True, data={"x": np.abs(inputs["x"])})
        """)
    check = _make_stage(tmp_path, "check", """
        from moe_data.stages.base import StageResult
        NAME = "check"
        def run(shape_id, inputs, cfg):
            return StageResult(True, meta={"min_x": float(inputs["x"].min()), "v": cfg["check"]["v"]})
        """)
    specs = [runner.StageSpec(raw, (), store_data=False), runner.StageSpec(clean, ()),
             runner.StageSpec(check, ("check",))]
    runner.run_chain("s", {}, specs, {"check": {"v": 1}}, tmp_path / "work")
    row = runner.run_chain("s", {}, specs, {"check": {"v": 2}}, tmp_path / "work")   # only "check" reruns
    assert row["check.min_x"] == 1.0                                                  # cleaned, not raw (-2)


def test_external_file_change_invalidates_stage(tmp_path, monkeypatch):
    """Rebuilding an external tool (e.g. the SphereMap binary) must invalidate the stage."""
    specs, counter = _setup(tmp_path, monkeypatch)
    tool = tmp_path / "tool.bin"
    tool.write_bytes(b"v1")
    specs = [runner.StageSpec(specs[0].module, (), external=("tool",))]
    cfg = {"paths": {"tool": str(tool)}}
    runner.run_chain("s1", {"x": np.arange(3.0)}, specs, cfg, tmp_path / "work")
    runner.run_chain("s1", {"x": np.arange(3.0)}, specs, cfg, tmp_path / "work")
    tool.write_bytes(b"v2")
    runner.run_chain("s1", {"x": np.arange(3.0)}, specs, cfg, tmp_path / "work")
    assert _calls(counter) == ["double", "double"]


def test_root_key_depends_on_input_identity(tmp_path):
    f = tmp_path / "mesh.npz"
    f.write_bytes(b"aaaa")
    k1 = runner.root_key({"file_path": str(f), "licence": "CC-BY"})
    assert k1 == runner.root_key({"file_path": str(f), "licence": "CC-BY"})
    assert k1 != runner.root_key({"file_path": str(f), "licence": "CC-BY-NC"})       # metadata change
    f.write_bytes(b"aaaaa")
    assert k1 != runner.root_key({"file_path": str(f), "licence": "CC-BY"})           # file changed


def test_crash_is_not_cached_and_retried(tmp_path, monkeypatch):
    """A transient crash (OOM, timeout) must be retried on the next run."""
    _setup(tmp_path, monkeypatch)
    flag = tmp_path / "fail_once"
    flag.write_text("x")
    flaky = _make_stage(tmp_path, "flaky", f"""
        import os
        from moe_data.stages.base import StageResult
        NAME = "flaky"
        def run(shape_id, inputs, cfg):
            if os.path.exists({str(flag)!r}):
                os.remove({str(flag)!r})
                raise MemoryError("transient")
            return StageResult(True)
        """)
    specs = [runner.StageSpec(flaky, ())]
    r1 = runner.run_chain("s1", {}, specs, {}, tmp_path / "work")
    r2 = runner.run_chain("s1", {}, specs, {}, tmp_path / "work")
    assert not r1["ok"] and r1["reject_reason"].startswith("crash: MemoryError")
    assert r2["ok"]
