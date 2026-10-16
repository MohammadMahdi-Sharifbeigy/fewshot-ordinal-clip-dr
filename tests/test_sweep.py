import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("sweep", Path("scripts/sweep.py"))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_build_command_keeps_override_with_space_intact():
    override = r"data.manifest=E:\Research\Dr.Mohammad(LAB)\ordinal classify\MMRDR\x.csv"
    cmd = sweep.build_command("configs/direction1/d1_full.yaml", 4, 2, [override, "train.epochs=3"])
    assert cmd[-2:] == [override, "train.epochs=3"]
    assert "data.k_shot=4" in cmd and "data.seed=2" in cmd


def test_extra_parses_one_token_per_override():
    args = sweep.parse_args(["--config", "c.yaml", "--extra", "a.b=E:/x y/z.csv", "c=1"])
    assert args.extra == ["a.b=E:/x y/z.csv", "c=1"]
