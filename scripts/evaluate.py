import os
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl

from src.data.datamodule import FewShotDataModule
from src.models.runner import FewShotOrdinalRunner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides (e.g. data.k_shot=4)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Same loading logic
    base_cfg = OmegaConf.load(args.config)
    if hasattr(base_cfg, 'defaults'):
        defaults_cfg = OmegaConf.load(os.path.join(os.path.dirname(args.config), base_cfg.defaults[0] + ".yaml"))
        cfg = OmegaConf.merge(defaults_cfg, base_cfg)
    else:
        cfg = base_cfg
        
    if args.overrides:
        override_cfg = OmegaConf.from_cli(args.overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
        
    # DataModule
    datamodule = FewShotDataModule(cfg)
    
    # Model
    model = FewShotOrdinalRunner(cfg)
    
    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=False
    )
    
    if args.test:
        trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
