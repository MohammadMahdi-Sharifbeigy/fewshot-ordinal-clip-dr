import argparse
from pathlib import Path

import pytorch_lightning as pl

from src.data.datamodule import FewShotDataModule
from src.models.runner import FewShotOrdinalRunner
from scripts.train import load_config  # reuse the shared loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Few-Shot Ordinal DR checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--ckpt",   type=str, default=None,   help="Path to .ckpt file (optional)")
    parser.add_argument("--test",   action="store_true",       help="Run on the test set")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf overrides, e.g. data.k_shot=4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    datamodule = FewShotDataModule(cfg)

    if args.ckpt:
        model = FewShotOrdinalRunner.load_from_checkpoint(args.ckpt, cfg=cfg)
    else:
        model = FewShotOrdinalRunner(cfg)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=False,
    )

    if args.test:
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
