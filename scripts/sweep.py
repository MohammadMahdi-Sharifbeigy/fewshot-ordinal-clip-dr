import subprocess
import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep over k-shot values and seeds using train.py"
    )
    parser.add_argument("--config",  type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--kshots", type=str, default="1,2,4,8,16,32",
        help="Comma-separated list of k values (default: 1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--seeds",  type=str, default="1,2,3,4,5",
        help="Comma-separated list of seeds (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--extra", nargs="*", default=[],
        help="Extra OmegaConf overrides appended to every run, one argument each "
             "(quote any that contain spaces), e.g. "
             "--extra \"data.manifest=/abs/CFP_manifest.csv\" \"data.splits_dir=/abs/splits\"",
    )
    return parser.parse_args(argv)


def build_command(config: str, k: int, seed: int, extra: list[str]) -> list[str]:
    """train.py command for one (k, seed) run; each override stays a single argv token."""
    return [
        sys.executable, "scripts/train.py",
        "--config", config,
        f"data.k_shot={k}",
        f"data.seed={seed}",
        *extra,
    ]


def main() -> None:
    args = parse_args()

    k_shots = [int(x) for x in args.kshots.split(",")]
    seeds   = [int(x) for x in args.seeds.split(",")]

    total = len(k_shots) * len(seeds)
    done  = 0

    for k in k_shots:
        for s in seeds:
            done += 1
            print(f"\n[{done}/{total}] config={args.config}  k={k}  seed={s}")
            cmd = build_command(args.config, k, s, args.extra)
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"[!] Run failed (k={k}, seed={s}) — continuing sweep.")

    print("\n[✓] Sweep complete.")


if __name__ == "__main__":
    main()
