"""
Multi-Modal Few-Shot DataModule
================================
Wraps MultiModalMMRDRDataset for PyTorch Lightning.

Key design decisions:
- All three modality manifests are constructed from cfg paths.
- Transforms are shared across modalities (same CLIP preprocessing).
- Support / query splitting uses the CFP split file as the anchor;
  OCT and UWF images for those patients are joined automatically.
- pin_memory + persistent_workers enabled for GPU throughput.
"""

import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from .multimodal_dataset import MultiModalMMRDRDataset

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_transform(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """CLIP-style preprocessing pipeline (same as existing unimodal code)."""
    load_size = image_size + 32
    if is_train:
        return transforms.Compose([
            transforms.Resize(load_size, interpolation=BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
        ])
    return transforms.Compose([
        transforms.Resize(load_size, interpolation=BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
    ])


class MultiModalFewShotDataModule(pl.LightningDataModule):
    """Lightning DataModule for few-shot multi-modal MMRDR.

    cfg.data must contain:
        manifests      : dict  {CFP: path, OCT: path, UWF: path}
        splits_dir     : str   directory containing split txt files
        k_shot         : int
        seed           : int
        batch_query    : int   batch size for query / val / test
        image_size     : int   (default 224)
        num_workers    : int   (default 4)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.k_shot = cfg.data.k_shot
        self.seed = cfg.data.seed
        self.batch_query = cfg.data.batch_query
        self.image_size = cfg.data.get("image_size", 224)
        self.num_workers = cfg.data.get("num_workers", 4)
        self.manifests = dict(cfg.data.manifests)  # {CFP: path, OCT: path, UWF: path}
        self.splits_dir = cfg.data.splits_dir

        # Build support/query split file paths (anchored on CFP)
        if self.k_shot <= 0:
            self.support_file: Optional[str] = None
            self.query_file: Optional[str] = None
        else:
            self.support_file = os.path.join(
                self.splits_dir,
                f"cfp_{self.k_shot}shot_s{self.seed}_support.txt",
            )
            self.query_file = os.path.join(
                self.splits_dir,
                f"cfp_{self.k_shot}shot_s{self.seed}_query.txt",
            )

    def setup(self, stage=None) -> None:
        train_tfm = get_transform(self.image_size, is_train=True)
        test_tfm = get_transform(self.image_size, is_train=False)

        # Apply same transform to all modalities (can be made per-modality if needed)
        train_transforms = {"CFP": train_tfm, "OCT": train_tfm, "UWF": train_tfm}
        test_transforms = {"CFP": test_tfm, "OCT": test_tfm, "UWF": test_tfm}

        if self.k_shot > 0:
            self.support_dataset = MultiModalMMRDRDataset(
                self.manifests,
                split_file=self.support_file,
                transforms=train_transforms,
            )
            self.query_dataset = MultiModalMMRDRDataset(
                self.manifests,
                split_file=self.query_file,
                transforms=test_transforms,
            )
        elif self.k_shot == -1:
            self.support_dataset = MultiModalMMRDRDataset(
                self.manifests,
                split_type="train",
                transforms=train_transforms,
            )
            self.query_dataset = MultiModalMMRDRDataset(
                self.manifests,
                split_type="test",
                transforms=test_transforms,
            )
        else:
            self.support_dataset = None
            self.query_dataset = None

        self.test_dataset = MultiModalMMRDRDataset(
            self.manifests,
            split_type="test",
            transforms=test_transforms,
        )

    def _make_loader(self, dataset, *, shuffle: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=(self.num_workers > 0),
            persistent_workers=(self.num_workers > 0),
            drop_last=False,
        )

    def train_dataloader(self) -> Optional[DataLoader]:
        if self.support_dataset is None or len(self.support_dataset) == 0:
            return None
        if self.k_shot == -1:
            batch_size = self.batch_query
        else:
            batch_size = min(len(self.support_dataset), self.batch_query)
        return self._make_loader(self.support_dataset, shuffle=True, batch_size=batch_size)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.query_dataset is None:
            return None
        return self._make_loader(self.query_dataset, shuffle=False, batch_size=self.batch_query)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=self.batch_query)
