import copy
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from .mmrdr_dataset import MMRDRDataset

# Standard CLIP normalization constants
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def get_transform(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Return standard CLIP-style image transform.

    Training: Resize to image_size+32 → RandomCrop → RandomHorizontalFlip → Normalize
    Eval:     Resize to image_size+32 → CenterCrop → Normalize

    The two-step Resize→Crop (rather than Resize to the exact target) ensures
    RandomCrop actually introduces positional jitter during training.
    """
    load_size = image_size + 32  # e.g. 256 for a 224 target
    if is_train:
        return transforms.Compose([
            transforms.Resize(load_size, interpolation=BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(load_size, interpolation=BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
        ])


class FewShotDataModule(pl.LightningDataModule):
    """Lightning DataModule for few-shot MMRDR evaluation.

    Support set  = k_shot images per grade (train DataLoader).
    Query set    = remaining training images (val DataLoader).
    Test set     = official held-out test split (test DataLoader).

    Full data (k_shot = -1): trains on the train split minus a stratified
    `data.full_val_fraction` holdout, which serves as the validation set.

    Batch size for the support DataLoader is automatically clamped to
    `len(support_dataset)` so that the loader is always valid regardless of
    whether k_shot is 1, 2, 4, 8, 16, or 32.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modality = cfg.data.modality
        self.k_shot = cfg.data.k_shot
        self.seed = cfg.data.seed
        self.batch_query = cfg.data.batch_query
        self.image_size = cfg.data.image_size
        self.manifest = cfg.data.manifest
        self.splits_dir = cfg.data.splits_dir
        self.num_workers = cfg.data.get("num_workers", 4)

        if self.k_shot <= 0:
            # Zero-shot (0) or Full-shot (-1): no support/query split files needed
            self.support_file = None
            self.query_file = None
        else:
            mod_lower = self.modality.lower()
            self.support_file = os.path.join(
                self.splits_dir,
                f"{mod_lower}_{self.k_shot}shot_s{self.seed}_support.txt",
            )
            self.query_file = os.path.join(
                self.splits_dir,
                f"{mod_lower}_{self.k_shot}shot_s{self.seed}_query.txt",
            )

    def setup(self, stage=None) -> None:
        train_transform = get_transform(self.image_size, is_train=True)
        test_transform  = get_transform(self.image_size, is_train=False)

        if self.k_shot > 0:
            self.support_dataset = MMRDRDataset(
                self.manifest, split_file=self.support_file, transform=train_transform
            )
            self.query_dataset = MMRDRDataset(
                self.manifest, split_file=self.query_file, transform=test_transform
            )
        elif self.k_shot == -1:
            # Full data: hold out a stratified part of the TRAIN split for
            # validation (early stopping / checkpoint selection). The test
            # split must never be used for model selection.
            train_paths, val_paths = self._full_data_holdout()
            self.support_dataset = MMRDRDataset(
                self.manifest, paths=train_paths, transform=train_transform
            )
            self.query_dataset = MMRDRDataset(
                self.manifest, paths=val_paths, transform=test_transform
            )
        else:
            self.support_dataset = None
            self.query_dataset = None

        # Official held-out test set (used in all settings, including zero-shot)
        self.test_dataset = MMRDRDataset(
            self.manifest, split_type="test", transform=test_transform
        )

    def _full_data_holdout(self) -> tuple[list[str], list[str]]:
        """Stratified, seeded train/val split of the official train split."""
        frac = float(self.cfg.data.get("full_val_fraction", 0.2))
        df = pd.read_csv(self.manifest)
        df = df[df["split"] == "train"]
        rng = np.random.default_rng(self.seed)
        val_paths = []
        for _, grade_df in df.groupby("grade"):
            n_val = max(1, int(round(len(grade_df) * frac)))
            val_paths.extend(rng.choice(grade_df["path"].to_numpy(), size=n_val, replace=False).tolist())
        val_set = set(val_paths)
        train_paths = [p for p in df["path"] if p not in val_set]
        return train_paths, val_paths

    def _make_loader(self, dataset, *, shuffle: bool, batch_size: int) -> DataLoader:
        """Helper to construct a DataLoader with consistent settings."""
        use_workers = self.num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=use_workers,
            pin_memory=(use_workers > 0),
            persistent_workers=(use_workers > 0),
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader | None:
        if self.support_dataset is None or len(self.support_dataset) == 0:
            return None
        # Clamp batch_size to the support set size.
        # With k_shot examples per grade: support size = k_shot * num_grades.
        # For k=1 this may be as small as 5; we must never request more than we have.
        if self.k_shot == -1:
            batch_size = self.batch_query
        else:
            batch_size = min(len(self.support_dataset), self.batch_query)
        return self._make_loader(self.support_dataset, shuffle=True, batch_size=batch_size)

    def support_eval_dataloader(self) -> DataLoader | None:
        """Support set with the test transform and fixed order (no augmentation).

        Used for statistics that must come from clean support images, e.g. CLAP's
        zero-shot λ initialisation.
        """
        if self.support_dataset is None or len(self.support_dataset) == 0:
            return None
        clean = copy.copy(self.support_dataset)
        clean.transform = get_transform(self.image_size, is_train=False)
        return self._make_loader(clean, shuffle=False, batch_size=self.batch_query)

    def val_dataloader(self) -> DataLoader | None:
        if self.query_dataset is None:
            return None
        return self._make_loader(self.query_dataset, shuffle=False, batch_size=self.batch_query)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=self.batch_query)
