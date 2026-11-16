import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from package_release import public_manifest, shard  # noqa: E402


def test_public_manifest_drops_licence_rows_and_private_columns():
    df = pd.DataFrame({"id": ["a", "b"], "ok": [True, False], "reject_stage": ["", "licence"],
                       "traceback": ["x", "y"], "original_file": ["data/raw/a.npz", ""], "author": ["u", "v"]})
    out = public_manifest(df)
    assert out["id"].tolist() == ["a"]
    assert "traceback" not in out and "original_file" not in out and "author" in out


def test_public_manifest_refuses_local_paths():
    df = pd.DataFrame({"id": ["a"], "ok": [True], "reject_stage": [""], "note": ["/Users/someone/x.npz"]})
    with pytest.raises(ValueError, match="local paths"):
        public_manifest(df)


def test_shards_respect_size_limit(tmp_path):
    files = []
    for i in range(7):
        f = tmp_path / f"s{i}.npz"
        f.write_bytes(b"x" * 400)
        files.append(f)
    groups = shard(files, 1000)
    assert [len(g) for g in groups] == [2, 2, 2, 1]
    assert sum(groups, []) == files
