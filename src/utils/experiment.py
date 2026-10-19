"""Run one experiment, with a result cache shared by every notebook and branch.

Runs are stored under one folder (by default `<project>/results_shared/`, next to
`data/`), keyed by a hash of the run's fully resolved config, not by its name:

    results_shared/<key>/config.yaml               resolved config (once per key)
    results_shared/<key>/names.txt                 every name this config was run under
    results_shared/<key>/CFP_<k>shot_s<seed>/      best.ckpt, last.ckpt, test_stats.json, geometry.json

Two notebooks that train the same configuration (e.g. "Ours (Full)" in the main
notebook and "Ours (master)" in a direction notebook) therefore share one run.
The key ignores only settings that do not change the result: where the data
lives, the number of loader workers, and k / seed (which name the sub-folder).

The key does not see the code. After changing model or training code in a way
that changes results, raise CACHE_VERSION so old runs are not reused.

Training follows the main notebook: validation every epoch, `best.ckpt` (best
val_qwk) and `last.ckpt` after every epoch, resume from `last.ckpt`, test with
the best checkpoint, k = 0 tests the untrained model.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.auto import tqdm

CACHE_VERSION = 1

# Settings that never change a run's result (k and seed name the sub-folder)
_IGNORED_KEYS = {"manifest", "splits_dir", "num_workers", "k_shot", "seed"}


def find_shared_results(start: Path, dataset: str = "DeepDRiD") -> Path:
    """`<project>/results_shared` of the first folder at or above `start` that holds data/<dataset>."""
    start = Path(start).resolve()
    for base in [start, *start.parents]:
        if (base / "data" / dataset / "CFP_manifest.csv").exists():
            return base / "results_shared"
    raise FileNotFoundError(f"No data/{dataset}/CFP_manifest.csv at or above {start}")


def load_run_config(config_path: str, k_shot: int, seed: int, overrides=(), data_dir=None):
    """Resolved config of one run. `data_dir` (absolute data/<dataset> folder) is for worktrees without data/."""
    from scripts.train import load_config

    base = [f"data.k_shot={k_shot}", f"data.seed={seed}"]
    if data_dir is not None:
        data_dir = Path(data_dir)
        base += [f"data.manifest={(data_dir / 'CFP_manifest.csv').as_posix()}",
                 f"data.splits_dir={(data_dir / 'splits').as_posix()}"]
    return load_config(config_path, base + list(overrides))


def config_key(cfg) -> str:
    """Hash of everything in the resolved config that can change the result."""
    data = OmegaConf.to_container(cfg, resolve=True)
    data["data"] = {k: v for k, v in data.get("data", {}).items() if k not in _IGNORED_KEYS}
    text = f"cache-v{CACHE_VERSION}\n" + json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def run_dir(results_root, cfg) -> Path:
    return Path(results_root) / config_key(cfg) / f"{cfg.data.modality}_{cfg.data.k_shot}shot_s{cfg.data.seed}"


def json_safe(value):
    """Metric values for JSON: numbers become floats, lists (e.g. confusion matrix) are kept."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, int)):
        return value
    return float(value)


def _register(results_root, cfg, name: str) -> None:
    key_dir = Path(results_root) / config_key(cfg)
    key_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = key_dir / "config.yaml"
    if not cfg_file.exists():
        OmegaConf.save(cfg, cfg_file)
    names_file = key_dir / "names.txt"
    names = names_file.read_text(encoding="utf-8").splitlines() if names_file.exists() else []
    if name not in names:
        with open(names_file, "a", encoding="utf-8") as f:
            f.write(name + "\n")


class EpochBar(pl.Callback):
    """Plain tqdm bars for one run: one tick per epoch (loss / val QWK / best val QWK),
    and below it one tick per batch of the current epoch (training, then validation)."""

    def __init__(self, desc: str, total: int):
        self.desc, self.total, self.bar, self.best, self.batch_bar = desc, total, None, None, None

    def on_train_start(self, trainer, pl_module):
        # A resumed run starts from the checkpoint's epoch and best score
        score = trainer.checkpoint_callback.best_model_score if trainer.checkpoint_callback else None
        self.best = float(score) if score is not None else None
        self.bar = tqdm(total=self.total, initial=trainer.current_epoch, desc=self.desc,
                        leave=False, unit="epoch", dynamic_ncols=True)

    def on_train_epoch_start(self, trainer, pl_module):
        n_val = sum(trainer.num_val_batches) if trainer.num_val_batches else 0
        self.batch_bar = tqdm(total=trainer.num_training_batches + n_val, desc=f"  epoch {trainer.current_epoch + 1}",
                              leave=False, unit="batch", dynamic_ncols=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.batch_bar is not None:
            self.batch_bar.set_postfix_str("train", refresh=False)
            self.batch_bar.update(1)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Skip Lightning's sanity check, which runs before the first epoch
        if self.batch_bar is not None and not trainer.sanity_checking:
            self.batch_bar.set_postfix_str("val", refresh=False)
            self.batch_bar.update(1)

    def on_train_epoch_end(self, trainer, pl_module):
        # Lightning runs ModelCheckpoint after other callbacks, so track the best here
        m = trainer.callback_metrics
        post = {key: f"{float(m[src]):.3f}" for key, src in (("loss", "train_loss"), ("val_qwk", "val_qwk")) if src in m}
        if "val_qwk" in m:
            self.best = max(self.best if self.best is not None else -1.0, float(m["val_qwk"]))
        if self.best is not None:
            post["best"] = f"{self.best:.3f}"
        self.bar.set_postfix(post, refresh=False)
        self.bar.update(1)
        if self.batch_bar is not None:
            self.batch_bar.close()
            self.batch_bar = None

    def on_train_end(self, trainer, pl_module):
        for bar in (self.batch_bar, self.bar):
            if bar is not None:
                bar.close()


def cached_metrics(config_path, k_shot, seed, results_root, overrides=(), data_dir=None):
    """Metrics of a finished run, or None (never trains)."""
    cfg = load_run_config(config_path, k_shot, seed, overrides, data_dir)
    stats = run_dir(results_root, cfg) / "test_stats.json"
    return json.load(open(stats)) if stats.exists() else None


def run_experiment(name: str, config_path: str, k_shot: int, seed: int, results_root,
                   overrides=(), data_dir=None, force_retrain: bool = False) -> dict:
    """Train (or load from the shared cache) one run and return its test metrics."""
    from src.data.datamodule import FewShotDataModule
    from src.models.runner import FewShotOrdinalRunner

    cfg = load_run_config(config_path, k_shot, seed, overrides, data_dir)
    out_dir = run_dir(results_root, cfg)
    _register(results_root, cfg, name)
    stats_file = out_dir / "test_stats.json"
    if stats_file.exists() and not force_retrain:
        return json.load(open(stats_file))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-run prints (seed, parameter count, anchor weights) would break the progress bars
    with contextlib.redirect_stdout(io.StringIO()):
        pl.seed_everything(seed, workers=True, verbose=False)
        datamodule = FewShotDataModule(cfg)
        model = FewShotOrdinalRunner(cfg)

    monitor = cfg.train.get("early_stopping_monitor", "val_qwk")
    mode = cfg.train.get("early_stopping_mode", "max")
    checkpoint = ModelCheckpoint(dirpath=str(out_dir), filename="best", monitor=monitor, mode=mode,
                                 save_top_k=1, save_last=True)
    callbacks = [EarlyStopping(monitor=monitor, patience=cfg.train.get("early_stopping_patience", 20), mode=mode),
                 checkpoint, EpochBar(f"{name} | k={k_shot} seed={seed}", cfg.train.epochs)]
    precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs if k_shot != 0 else 0,
        accelerator="auto", devices=1, precision=precision, gradient_clip_val=1.0,
        callbacks=callbacks, log_every_n_steps=1, enable_checkpointing=True, logger=False,
        enable_progress_bar=False, enable_model_summary=False,
    )

    geometry = []
    with contextlib.redirect_stdout(io.StringIO()):
        if k_shot != 0:
            last_ckpt = out_dir / "last.ckpt"
            trainer.fit(model, datamodule=datamodule,
                        ckpt_path=str(last_ckpt) if last_ckpt.exists() else None, weights_only=False)
            geometry = list(getattr(model, "geometry_history", []))
            # Test with the best checkpoint
            if checkpoint.best_model_path:
                state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
                model.load_state_dict(state["state_dict"])
        trainer.test(model, datamodule=datamodule, verbose=False)

    metrics = json_safe(model.latest_metrics)
    with open(stats_file, "w") as f:
        json.dump(metrics, f, indent=4)
    if geometry:
        with open(out_dir / "geometry.json", "w") as f:
            json.dump(json_safe(geometry), f, indent=2)
    return metrics


def find_runs(results_root, names, modality: str = "CFP") -> pd.DataFrame:
    """Finished runs recorded under any of `names` (one row per k and seed), without training.

    Reads names.txt of every cached config, so it also finds runs made from
    configs that do not exist on the current branch. If one name was used for
    several configs, only the most recently registered config is kept.
    """
    names = [names] if isinstance(names, str) else list(names)
    rows = []
    for name in names:
        keys = [p.parent for p in Path(results_root).glob("*/names.txt")
                if name in p.read_text(encoding="utf-8").splitlines()]
        if not keys:
            continue
        if len(keys) > 1:
            print(f"'{name}' was run with {len(keys)} different configs; using the most recent one")
        key_dir = max(keys, key=lambda p: (p / "names.txt").stat().st_mtime)
        for stats in sorted(key_dir.glob(f"{modality}_*shot_s*/test_stats.json")):
            k, seed = stats.parent.name.split("_")[1].replace("shot", ""), stats.parent.name.split("_s")[-1]
            rows.append({"Variant": name, "k": int(k), "Seed": int(seed), "config_key": key_dir.name,
                         **json.load(open(stats))})
    return pd.DataFrame(rows)
