"""
prepare_external_datasets.py
────────────────────────────
One-time script that converts the raw APTOS and DeepDRiD directories into the
standardised CFP_manifest.csv format expected by FewShotDataModule / MMRDRDataset.

Expected raw layout
───────────────────
APTOS (F:/Datasets/DR/APTOS):
  train_images/train_images/<id_code>.png
  train_1.csv   – columns: id_code, diagnosis       (train split)
  valid.csv     – columns: id_code, diagnosis        (val/test split)

DeepDRiD (F:/Datasets/DR/DeepDRiD/DeepDRiD-master):
  regular_fundus_images/Images/<patient_id>/<image_id>.jpg   (both splits)
  regular-fundus-training.csv   – columns: image_id, patient_id, left/right_eye_DR_Level, ...
  regular-fundus-validation.csv – columns: image_id, patient_id, left/right_eye_DR_Level, ...

Output
──────
data/APTOS/CFP_manifest.csv
data/DeepDRiD/CFP_manifest.csv

Both files have columns:  path, modality, grade, lesion, lr, split
  path     – path relative to the dataset root (see src/data/paths.py)
  modality – "CFP"
  grade    – DR grade (0-4); DeepDRiD: grade of the imaged eye
  lesion   – "" (not available in these datasets)
  lr       – DeepDRiD: 0 = left eye, 1 = right eye; APTOS: 0 (not available)
  split    – "train" or "test"
DeepDRiD also has: patient (id) and patient_grade (worse eye, for reference)

Run from the Workspace directory:
  python scripts/prepare_external_datasets.py
or with custom paths:
  python scripts/prepare_external_datasets.py --aptos F:/Datasets/DR/APTOS
                                               --deepdrid F:/Datasets/DR/DeepDRiD/DeepDRiD-master
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ── Defaults ─────────────────────────────────────────────────────────────────
# Dataset roots live in src/data/paths.py (Linux path first, then Windows), so
# the loaders resolve the relative manifest paths against the same folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.paths import DEFAULT_APTOS, DEFAULT_DEEPDRID  # noqa: E402


def build_aptos_manifest(aptos_root: Path, out_dir: Path) -> Path:
    """Convert raw APTOS CSV files -> CFP_manifest.csv"""
    aptos_root = Path(aptos_root)
    # Each CSV has its own image folder; paths are stored relative to aptos_root
    img_dirs = {
        "train_1.csv": "train_images/train_images",
        "valid.csv":   "val_images/val_images",
        "val.csv":     "val_images/val_images",
        "test.csv":    "test_images/test_images",
    }

    rows = []

    # ── Training split ────────────────────────────────────────────────────────
    train_csv = aptos_root / "train_1.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        for _, row in df.iterrows():
            rows.append({
                "path":     f"{img_dirs['train_1.csv']}/{row['id_code']}.png",
                "modality": "CFP",
                "grade":    int(row["diagnosis"]),
                "lesion":   "",
                "lr":       0,
                "split":    "train",
            })
        print(f"  [APTOS] train_1.csv  -> {len(df):,} rows")
    else:
        print(f"  [APTOS] WARNING: {train_csv} not found")

    # ── Validation / test split ───────────────────────────────────────────────
    for val_name in ("valid.csv", "test.csv", "val.csv"):
        val_csv = aptos_root / val_name
        if val_csv.exists():
            df = pd.read_csv(val_csv)
            for _, row in df.iterrows():
                rows.append({
                    "path":     f"{img_dirs[val_name]}/{row['id_code']}.png",
                    "modality": "CFP",
                    "grade":    int(row["diagnosis"]),
                    "lesion":   "",
                    "lr":       0,
                    "split":    "test",
                })
            print(f"  [APTOS] {val_name}  -> {len(df):,} rows (test split)")
            break  # use only the first found
    else:
        print("  [APTOS] WARNING: no valid/test CSV found — only train split will be saved")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "CFP_manifest.csv"
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  [APTOS] Manifest saved -> {out_path}")
    print(f"  Rows: {len(manifest):,}")
    if not manifest.empty:
        print(f"  Grade distribution:\n{manifest['grade'].value_counts().sort_index()}")
        missing = manifest[~manifest['path'].apply(lambda p: (aptos_root / p).exists())]
        print(f"  Missing files: {len(missing):,} / {len(manifest):,}")

    return out_path


def build_deepdrid_manifest(deepdrid_root: Path, out_dir: Path) -> Path:
    """Convert raw DeepDRiD CSV files -> CFP_manifest.csv

    Actual layout inside DeepDRiD-master/regular_fundus_images/:
      regular-fundus-training/
        regular-fundus-training.csv
        Images/<patient_id>/<image_id>.jpg
      regular-fundus-validation/
        regular-fundus-validation.csv
        Images/<patient_id>/<image_id>.jpg
    """
    deepdrid_root = Path(deepdrid_root)
    base = deepdrid_root / "regular_fundus_images"

    rows = []

    split_map = {
        "regular-fundus-training":   "train",
        "regular-fundus-validation": "test",
    }

    for subdir, split_label in split_map.items():
        split_dir = base / subdir
        csv_name  = f"{subdir}.csv"
        csv_path  = split_dir / csv_name
        img_base  = split_dir / "Images"

        if not csv_path.exists():
            print(f"  [DeepDRiD] WARNING: {csv_path} not found -- skipping")
            continue

        df = pd.read_csv(csv_path)
        # Every row grades exactly one eye. A few training rows put the grade in
        # the other eye's column (e.g. 56_r3 has only left_eye_DR_Level), so take
        # whichever column is filled rather than trusting the l/r in the image id.
        left, right = df["left_eye_DR_Level"], df["right_eye_DR_Level"]
        if not (left.notna() ^ right.notna()).all():
            raise ValueError(f"{csv_path}: expected exactly one eye grade per row")
        eye_grade = left.fillna(right).astype(int)
        count = 0
        for (_, row), grade in zip(df.iterrows(), eye_grade):
            patient_id = str(int(row["patient_id"]))
            image_id   = str(row["image_id"])

            # Images/<patient_id>/<image_id>.jpg, stored relative to the
            # DeepDRiD root (forward slashes) so the manifest is portable
            img_path = img_base / patient_id / f"{image_id}.jpg"
            rows.append({
                "path":          img_path.relative_to(deepdrid_root).as_posix(),
                "modality":      "CFP",
                "grade":         int(grade),
                "lesion":        "",
                "lr":            0 if "_l" in image_id else 1,
                "split":         split_label,
                "patient":       patient_id,
                "patient_grade": int(row["patient_DR_Level"]),
            })
            count += 1
        print(f"  [DeepDRiD] {csv_name} -> {count:,} rows ({split_label} split)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "CFP_manifest.csv"
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_path, index=False)

    print(f"\n  [DeepDRiD] Manifest saved -> {out_path}")
    print(f"  Rows: {len(manifest):,}")
    if not manifest.empty:
        print(f"  Grade distribution:\n{manifest['grade'].value_counts().sort_index()}")
        missing = manifest[~manifest['path'].apply(lambda p: (deepdrid_root / p).exists())]
        print(f"  Missing files: {len(missing):,} / {len(manifest):,}")

    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare CFP_manifest.csv for APTOS and/or DeepDRiD datasets."
    )
    parser.add_argument("--aptos",    type=Path, default=DEFAULT_APTOS,
                        help="Root directory of raw APTOS dataset")
    parser.add_argument("--deepdrid", type=Path, default=DEFAULT_DEEPDRID,
                        help="Root directory of raw DeepDRiD-master dataset")
    parser.add_argument("--skip-aptos",    action="store_true")
    parser.add_argument("--skip-deepdrid", action="store_true")
    return parser.parse_args()


def main():
    # Locate workspace root (parent of 'configs/' dir)
    workspace = Path(__file__).resolve().parent.parent
    while not (workspace / "configs").exists():
        if workspace == workspace.parent:
            print("ERROR: Could not locate workspace root (no 'configs/' found).")
            sys.exit(1)
        workspace = workspace.parent

    args = parse_args()

    print(f"Workspace root : {workspace}")
    print("=" * 60)

    if not args.skip_aptos:
        print("\n[1/2] Building APTOS manifest …")
        if not args.aptos.exists():
            print(f"  WARNING: APTOS root not found at {args.aptos}")
        else:
            build_aptos_manifest(args.aptos, workspace / "data" / "APTOS")

    if not args.skip_deepdrid:
        print("\n[2/2] Building DeepDRiD manifest …")
        if not args.deepdrid.exists():
            print(f"  WARNING: DeepDRiD root not found at {args.deepdrid}")
        else:
            build_deepdrid_manifest(args.deepdrid, workspace / "data" / "DeepDRiD")

    print("\nDone.")


if __name__ == "__main__":
    main()
