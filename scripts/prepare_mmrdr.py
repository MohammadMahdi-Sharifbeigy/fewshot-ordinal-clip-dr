import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.paths import DEFAULT_MMRDR  # noqa: E402

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmrdr_root", type=str, default=str(DEFAULT_MMRDR), help="Path to MMRDR root directory")
    parser.add_argument("--modality", type=str, required=True, choices=["CFP", "OCT", "UWF", "ALL"], help="Modality to process")
    return parser.parse_args()

def process_modality(root, mod):
    print(f"Processing {mod}...")
    csv_path = os.path.join(root, f"MMRDR-{mod}", f"{'FP' if mod=='CFP' else mod}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotation file not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Paths relative to the MMRDR root (portable across machines), e.g.
    # 'MMRDR-CFP/img/tr000001.jpg'; csv image paths are like 'img/tr000001.jpg'
    df['path'] = df['image'].apply(lambda x: f"MMRDR-{mod}/{x}".replace('\\', '/'))
    missing = sum(not os.path.exists(os.path.join(root, p)) for p in df['path'])
    
    # Determine split: train or test
    df['split'] = df['image'].apply(lambda x: "train" if "img/tr" in x else "test")
    
    # Rename columns to standardized format
    df['modality'] = mod
    
    # Make sure we have standard columns
    cols = ['path', 'modality', 'grade', 'lesion', 'lr', 'split']
    # If lesion doesn't exist (e.g. OCT), fill with empty
    if 'lesion' not in df.columns:
        df['lesion'] = ''
        
    manifest = df[cols]
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "MMRDR")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{mod}_manifest.csv")
    manifest.to_csv(out_csv, index=False)
    print(f"Saved manifest for {mod} with {len(manifest)} rows to {out_csv}")
    print("Train count:", sum(manifest['split'] == 'train'))
    print("Test count:", sum(manifest['split'] == 'test'))
    print("Missing files:", missing)
    print()

def main():
    args = parse_args()
    if args.modality == "ALL":
        for mod in ["CFP", "OCT", "UWF"]:
            process_modality(args.mmrdr_root, mod)
    else:
        process_modality(args.mmrdr_root, args.modality)

if __name__ == "__main__":
    main()
