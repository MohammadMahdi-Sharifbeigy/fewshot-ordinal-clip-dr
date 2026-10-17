"""Write OrdinalCLIP-format DeepDRiD lists (`<relative path> <grade>`) from the manifest.

Paths are relative to <DeepDRiD root>/regular_fundus_images, the images_root
used by notebooks/OrdinalCLIP_DeepDRiD_FewShot_Simulation.ipynb.
"""
import sys
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
manifest_path = WORKSPACE / "data" / "DeepDRiD" / "CFP_manifest.csv"
out_dir = WORKSPACE / "data" / "DeepDRiD" / "ordinalclip_splits"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(manifest_path)

PREFIX = "regular_fundus_images/"

def to_relative(p):
    # Manifest paths are relative to the DeepDRiD root
    p = p.replace('\\', '/')
    if not p.startswith(PREFIX):
        sys.exit(f"Unexpected manifest path {p!r}; rebuild it with scripts/prepare_external_datasets.py")
    return p[len(PREFIX):]

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'test']  # The test split in DeepDRiD is used as validation in these simulations

with open(out_dir / 'train.txt', 'w', encoding='utf-8') as f:
    for _, row in train_df.iterrows():
        rel_path = to_relative(row['path'])
        f.write(f"{rel_path} {row['grade']}\n")

with open(out_dir / 'val.txt', 'w', encoding='utf-8') as f:
    for _, row in val_df.iterrows():
        rel_path = to_relative(row['path'])
        f.write(f"{rel_path} {row['grade']}\n")

print("Created train.txt and val.txt in", out_dir)
