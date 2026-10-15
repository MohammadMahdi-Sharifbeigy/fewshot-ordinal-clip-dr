import os
import pandas as pd
from pathlib import Path

manifest_path = r'd:\Research\Dr.Mohammad(LAB)\ordinal classify\MMRDR\Workspace\data\DeepDRiD\CFP_manifest.csv'
out_dir = Path(r'd:\Research\Dr.Mohammad(LAB)\ordinal classify\MMRDR\Workspace\data\DeepDRiD\ordinalclip_splits')
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(manifest_path)

images_root = r"F:\Datasets\DR\DeepDr\DeepDRiD-master\regular_fundus_images"

def to_relative(p):
    # Ensure standard slashes
    p = p.replace('\\', '/')
    root = images_root.replace('\\', '/')
    if p.lower().startswith(root.lower()):
        return p[len(root):].lstrip('/')
    return p

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
