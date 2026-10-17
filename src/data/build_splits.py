import os
import sys
import argparse
import pandas as pd

# Allow `python src/data/build_splits.py` from the workspace root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.splitting import few_shot_split, patient_ids  # noqa: E402

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="CFP", choices=["CFP", "OCT", "UWF"])
    parser.add_argument("--dataset", type=str, default="MMRDR",
                        help="Dataset name (MMRDR | APTOS | DeepDRiD). "
                             "Must match a folder under data/<dataset>/ containing CFP_manifest.csv")
    parser.add_argument("--kshots", type=str, default="1,2,4,8,16", help="Comma separated k values")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5", help="Comma separated seeds")
    return parser.parse_args()

def main():
    args = parse_args()
    k_shots = [int(x) for x in args.kshots.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    dataset = args.dataset
    modality = args.modality

    # Locate the manifest: first look for modality-specific, then CFP_manifest.csv
    data_dir = os.path.join(workspace_dir, "data", dataset)
    manifest_path = os.path.join(data_dir, f"{modality}_manifest.csv")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(data_dir, "CFP_manifest.csv")
    if not os.path.exists(manifest_path):
        prep_hint = (
            "prepare_mmrdr.py" if dataset == "MMRDR"
            else "scripts/prepare_external_datasets.py"
        )
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run {prep_hint} first."
        )

    df = pd.read_csv(manifest_path)

    # We only sample k-shot support from the training set.
    df_train = df[df['split'] == 'train'].reset_index(drop=True)

    out_dir = os.path.join(data_dir, "splits")
    os.makedirs(out_dir, exist_ok=True)

    mod_lower = modality.lower()

    def write(paths, name):
        # Paths are written exactly as in the manifest (relative for DeepDRiD)
        pd.Series(paths, dtype=str).to_csv(os.path.join(out_dir, name), index=False, header=False)

    # Supports are nested across k and independent of which k values are
    # requested; the query (validation) set excludes every support patient.
    for seed in seeds:
        for k in k_shots:
            support, query = few_shot_split(df_train, k, seed)
            write(support, f"{mod_lower}_{k}shot_s{seed}_support.txt")
            write(query, f"{mod_lower}_{k}shot_s{seed}_query.txt")
            print(f"Generated {mod_lower} {k}-shot seed {seed}: {len(support)} support, {len(query)} query")

        # Base-to-novel: support has 16 shots of grades 0-3, novel is grade 4;
        # query is every image of the patients not in the support
        base = df_train[df_train['grade'] <= 3]
        b2n_support, _ = few_shot_split(base, 16, seed)
        patients = patient_ids(df_train)
        support_patients = set(patients[df_train['path'].isin(set(b2n_support))])
        b2n_query = df_train.loc[~patients.isin(support_patients).to_numpy(), 'path'].tolist()
        write(b2n_support, f"{mod_lower}_b2n_s{seed}_support.txt")
        write(b2n_query, f"{mod_lower}_b2n_s{seed}_query.txt")

if __name__ == "__main__":
    main()
