import os
import argparse
import json
from omegaconf import OmegaConf
import pytorch_lightning as pl

from src.data.datamodule import FewShotDataModule
from src.models.runner import FewShotOrdinalRunner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides (e.g. data.k_shot=4)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    base_cfg = OmegaConf.load(args.config)
    
    # Check if there are defaults to load
    if hasattr(base_cfg, 'defaults'):
        defaults_cfg = OmegaConf.load(os.path.join(os.path.dirname(args.config), base_cfg.defaults[0] + ".yaml"))
        cfg = OmegaConf.merge(defaults_cfg, base_cfg)
    else:
        cfg = base_cfg
        
    if args.overrides:
        override_cfg = OmegaConf.from_cli(args.overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
        
    pl.seed_everything(cfg.data.seed)
    
    # DataModule
    datamodule = FewShotDataModule(cfg)
    
    # Model
    model = FewShotOrdinalRunner(cfg)
    
    # Output dir
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join("results", config_name, f"{cfg.data.modality}_{cfg.data.k_shot}shot_s{cfg.data.seed}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed", # Use mixed precision for CLIP
        enable_checkpointing=False, # Save space, only save stats
        logger=False # Disable tensorboard for simplicity
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    # Test
    trainer.test(model, datamodule=datamodule)
    
    # Save stats
    stats_file = os.path.join(out_dir, "test_stats.json")
    with open(stats_file, 'w') as f:
        json.dump({k: float(v) for k, v in model.latest_metrics.items()}, f, indent=4)
        
    print(f"Metrics saved to {stats_file}")

if __name__ == "__main__":
    main()
