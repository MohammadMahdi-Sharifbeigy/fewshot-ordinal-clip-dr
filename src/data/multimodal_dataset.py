"""
Multi-Modal MMRDR Dataset
=========================
Returns a dict of tensors (one per modality) so that all three modalities
(CFP, OCT, UWF) for the same patient index can be batched together cleanly.

Key design decisions for speed:
- pandas is used only at init time; __getitem__ is pure numpy/PIL/torch.
- Missing modalities are returned as torch.zeros (not None) so DataLoader
  can stack them without a custom collate_fn.
- A boolean missing_mask tensor tells the model which modalities are absent.
"""

import os
from typing import Optional, Dict, Tuple

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


MODALITIES = ("CFP", "OCT", "UWF")
# OCT has 3 grades (0-2); CFP and UWF have 5 grades (0-4)
MODALITY_NUM_RANKS = {"CFP": 5, "OCT": 3, "UWF": 5}


class MultiModalMMRDRDataset(Dataset):
    """Patient-aligned multi-modal dataset for MMRDR.

    Each index corresponds to one patient (shared ID derived from filename
    stem, e.g. ``tr000001``).  The manifests are joined on this stem.

    Args:
        manifests:   Dict mapping modality name → path to manifest CSV.
        split_file:  Optional text file listing paths (for few-shot support set).
        split_type:  ``"train"`` or ``"test"`` — uses the manifest's own split column.
        transforms:  Dict mapping modality name → torchvision transform.
        missing_fill: If ``True``, a black image tensor is returned for any
                      modality a patient is missing; ``missing_mask`` will be
                      ``False`` for that slot.
    """

    def __init__(
        self,
        manifests: Dict[str, str],
        split_file: Optional[str] = None,
        split_type: Optional[str] = None,
        transforms: Optional[Dict] = None,
        missing_fill: bool = True,
    ):
        super().__init__()
        self.transforms = transforms or {}
        self.missing_fill = missing_fill

        # Load and align all three manifests
        dfs = {}
        for mod, csv_path in manifests.items():
            df = pd.read_csv(csv_path)
            # Extract patient stem from path (e.g. "tr000001" from the filename)
            df["_stem"] = df["path"].apply(
                lambda p: os.path.splitext(os.path.basename(p))[0]
            )
            dfs[mod] = df.set_index("_stem")

        # Apply split filter using CFP manifest as the anchor
        anchor_df = dfs["CFP"]
        if split_file is not None:
            with open(split_file, "r") as f:
                valid_paths = {line.strip() for line in f if line.strip()}
            anchor_df = anchor_df[anchor_df["path"].isin(valid_paths)]
        elif split_type is not None:
            anchor_df = anchor_df[anchor_df["split"] == split_type]

        self.patient_stems = anchor_df.index.tolist()
        self.dfs = dfs  # {modality: df indexed by stem}

        # Pre-build fast lookup arrays (avoids repeated .loc in __getitem__)
        # Store as dicts of numpy arrays for O(1) row access.
        self._build_lookup()

    def _build_lookup(self):
        """Build stem → row-index mapping for each modality for fast access."""
        self._lookup: Dict[str, Dict[str, int]] = {}
        for mod, df in self.dfs.items():
            self._lookup[mod] = {
                stem: i for i, stem in enumerate(df.index)
            }
        # Store numpy arrays for path and grade per modality
        self._paths: Dict[str, np.ndarray] = {
            mod: df["path"].to_numpy() for mod, df in self.dfs.items()
        }
        self._grades: Dict[str, np.ndarray] = {
            mod: df["grade"].to_numpy() for mod, df in self.dfs.items()
        }

    def __len__(self) -> int:
        return len(self.patient_stems)

    def _load_image(self, path: str, modality: str) -> torch.Tensor:
        """Load a single image and apply the modality-specific transform."""
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return black image on failure — never crash the DataLoader
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        tfm = self.transforms.get(modality)
        if tfm is not None:
            return tfm(img)
        return torch.zeros(3, 224, 224, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict:
        stem = self.patient_stems[idx]

        images: Dict[str, torch.Tensor] = {}
        grades: Dict[str, int] = {}
        missing: Dict[str, bool] = {}

        for mod in MODALITIES:
            lookup = self._lookup[mod]
            if stem in lookup:
                row_idx = lookup[stem]
                path = self._paths[mod][row_idx]
                grade = int(self._grades[mod][row_idx])
                images[mod] = self._load_image(path, mod)
                grades[mod] = grade
                missing[mod] = False
            else:
                # Modality missing for this patient
                images[mod] = torch.zeros(3, 224, 224, dtype=torch.float32)
                grades[mod] = -1  # sentinel
                missing[mod] = True

        # CFP grade is the canonical patient-level label (0-4)
        patient_grade = grades["CFP"] if not missing["CFP"] else max(
            g for g in grades.values() if g >= 0
        )

        # Stack images into [M, C, H, W] and missing mask into [M]
        image_stack = torch.stack([images[m] for m in MODALITIES], dim=0)  # [3, C, H, W]
        missing_mask = torch.tensor([missing[m] for m in MODALITIES], dtype=torch.bool)  # [3]
        grade_tensor = torch.tensor(
            [grades[m] for m in MODALITIES], dtype=torch.long
        )  # [3]  (-1 for missing)
        patient_grade_tensor = torch.tensor(patient_grade, dtype=torch.long)

        return {
            "images": image_stack,          # [3, C, H, W]
            "grades": grade_tensor,          # [3]  per-modality grades (-1 = missing)
            "patient_grade": patient_grade_tensor,  # scalar
            "missing": missing_mask,         # [3] bool (True = absent)
            "stem": stem,
        }
