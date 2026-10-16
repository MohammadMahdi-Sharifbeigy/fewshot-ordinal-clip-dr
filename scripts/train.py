import os
import json
import argparse
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.data.datamodule import FewShotDataModule
from src.models.runner import FewShotOrdinalRunner


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _resolve_default(base_path: Path, default_ref: str) -> Path:
    """Resolve a 'defaults' reference relative to the config file's directory.

    Handles both same-directory (e.g. 'default') and parent-directory
    references (e.g. '../default') that appear in baseline configs.
    """
    resolved = (base_path.parent / (default_ref + ".yaml")).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Default config not found: {resolved}\n"
            f"  (referenced as '{default_ref}' from {base_path})"
        )
    return resolved


def load_config(config_path: str, overrides: list[str] | None = None) -> DictConfig:
    """Load a (possibly nested) OmegaConf config with defaults chaining.

    Supports the pattern:
        defaults:
          - ../default   # or - default
    used in baseline configs under configs/baselines/.
    """
    path = Path(config_path).resolve()
    cfg = OmegaConf.load(path)

    if "defaults" in cfg:
        # Merge all defaults in order (each resolved recursively, so a default
        # may itself have defaults), then overlay the specific config on top
        merged = OmegaConf.create({})
        for default_ref in cfg.defaults:
            default_path = _resolve_default(path, default_ref)
            default_cfg = load_config(str(default_path))
            merged = OmegaConf.merge(merged, default_cfg)
        # Remove 'defaults' key before merging (it's metadata, not a model field)
        cfg_without_defaults = OmegaConf.masked_copy(
            cfg, [k for k in cfg if k != "defaults"]
        )
        cfg = OmegaConf.merge(merged, cfg_without_defaults)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(overrides))

    return cfg


def json_safe(value):
    """Metric values for JSON: numbers become floats, lists (e.g. confusion matrix) are kept."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, int)):
        return value
    return float(value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Few-Shot Ordinal DR model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-notation overrides, e.g. data.k_shot=4 train.lr=2e-4",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    pl.seed_everything(cfg.data.seed, workers=True)

    # ── DataModule & Model ─────────────────────────────────────────────────
    datamodule = FewShotDataModule(cfg)
    model = FewShotOrdinalRunner(cfg)

    # ── Output directory ───────────────────────────────────────────────────
    config_name = Path(args.config).stem
    out_dir = Path("results") / config_name / f"{cfg.data.modality}_{cfg.data.k_shot}shot_s{cfg.data.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Callbacks ──────────────────────────────────────────────────────────
    callbacks = []

    monitor_metric = cfg.train.get("early_stopping_monitor", "val_qwk")
    monitor_mode   = cfg.train.get("early_stopping_mode", "max")
    patience       = cfg.train.get("early_stopping_patience", 20)

    callbacks.append(
        EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            mode=monitor_mode,
            verbose=True,
        )
    )
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(out_dir),
            filename="best",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            verbose=False,
        )
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=callbacks,
        # Few-shot support sets are tiny; log every step so we don't miss anything
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=False,
    )

    # ── Train → Test ───────────────────────────────────────────────────────
    geometry_history = []
    if cfg.data.k_shot == 0:
        # Zero-shot: evaluate the model in its initial state, no training
        print("\n[*] k_shot = 0: skipping training, testing the untrained model")
    else:
        trainer.fit(model, datamodule=datamodule)
        geometry_history = list(model.geometry_history)

        # Load best checkpoint before testing
        best_ckpt = callbacks[1].best_model_path
        if best_ckpt:
            print(f"\n[*] Loading best checkpoint: {best_ckpt}")
            model = FewShotOrdinalRunner.load_from_checkpoint(best_ckpt, cfg=cfg)

    trainer.test(model, datamodule=datamodule)

    # ── Save metrics ───────────────────────────────────────────────────────
    stats_file = out_dir / "test_stats.json"
    with open(stats_file, "w") as f:
        json.dump(json_safe(model.latest_metrics), f, indent=4)

    if geometry_history:
        with open(out_dir / "geometry.json", "w") as f:
            json.dump(json_safe(geometry_history), f, indent=2)

    print(f"\n[✓] Metrics saved to {stats_file}")
    for k, v in model.latest_metrics.items():
        if isinstance(v, (list, tuple)):
            print(f"    {k:>6}: {v}")
        else:
            print(f"    {k:>6}: {v:.4f}")


if __name__ == "__main__":
    main()
