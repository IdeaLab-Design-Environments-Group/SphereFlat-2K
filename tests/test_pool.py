"""Regression: the worker pool must finish when there are many more jobs than workers.

With ProcessPoolExecutor(max_tasks_per_child=25) on Python 3.12 the pool deadlocked once every worker had
run 25 tasks (the Thingi10K survey stopped at exactly 12 x 25 = 300 shapes)."""
import pytest

from moe_data.run import _pool_map


def _square(sid, x):
    return {"id": sid, "ok": True, "y": x * x}


def _boom(sid, x):
    raise RuntimeError("worker failure")


@pytest.mark.timeout(60)
def test_pool_finishes_many_jobs():
    jobs = [(f"s{i}", i) for i in range(130)]               # > 2 workers x 25 tasks
    rows = _pool_map(_square, jobs, {"run": {"max_workers": 2}}, "t")
    assert sorted(r["y"] for r in rows) == [i * i for i in range(130)]


@pytest.mark.timeout(60)
def test_pool_records_worker_exception():
    rows = _pool_map(_boom, [("s0", 1)], {"run": {"max_workers": 1}}, "t")
    assert rows[0]["ok"] is False and rows[0]["reject_reason"].startswith("crash: RuntimeError")


def test_stage_health_aborts_on_mass_crashes():
    """A broken dependency (e.g. a library that loads no plugins) must stop the run, not silently produce an
    empty dataset of crash rejections."""
    from moe_data.run import StageHealthError, check_stage_health
    rows = [{"id": f"s{i}", "ok": False, "reject_stage": "remesh",
             "reject_reason": "crash: AttributeError: no attribute"} for i in range(30)]
    rows += [{"id": f"t{i}", "ok": True, "reject_stage": "", "reject_reason": ""} for i in range(10)]
    with pytest.raises(StageHealthError, match="remesh: 30 of 40"):
        check_stage_health(rows, max_crash_share=0.2)
    check_stage_health(rows[25:], max_crash_share=0.5)          # 5 of 15 crashed: below the limit


def test_preflight_detects_missing_binary(tmp_path):
    from moe_data.run import StageHealthError, preflight
    cfg = {"paths": {"spheremap_bin": str(tmp_path / "missing")}}
    with pytest.raises(StageHealthError, match="SphereMap binary"):
        preflight(cfg)
