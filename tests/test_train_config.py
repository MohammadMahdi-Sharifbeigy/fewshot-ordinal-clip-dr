import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("train", Path("scripts/train.py"))
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def test_load_config_recursive_defaults(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\ny: {p: 1, q: 2}\n")
    (tmp_path / "b.yaml").write_text("defaults:\n  - a\ny: {q: 3}\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.yaml").write_text("defaults:\n  - ../b\nz: 9\n")
    cfg = train.load_config(str(sub / "c.yaml"), ["y.p=7"])
    assert cfg.x == 1 and cfg.y.p == 7 and cfg.y.q == 3 and cfg.z == 9 and "defaults" not in cfg


def test_json_safe_stats_keeps_lists():
    out = train.json_safe({"qwk": 0.5, "confusion_matrix": [[1, 0], [0, 1]], "mean_mass_by_dist": [1.0, 0.0]})
    assert json.loads(json.dumps(out)) == {"qwk": 0.5, "confusion_matrix": [[1, 0], [0, 1]],
                                           "mean_mass_by_dist": [1.0, 0.0]}
