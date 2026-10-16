import os
import argparse
import pandas as pd
import numpy as np

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
    df_train = df[df['split'] == 'train']

    out_dir = os.path.join(data_dir, "splits")
    os.makedirs(out_dir, exist_ok=True)

    mod_lower = modality.lower()
    
    for seed in seeds:
        rng = np.random.default_rng(seed)
        
        for k in k_shots:
            support_indices = []
            
            # Stratified sampling
            for grade in sorted(df_train['grade'].unique()):
                grade_df = df_train[df_train['grade'] == grade]
                if len(grade_df) < k:
                    print(f"Warning: Not enough samples for grade {grade} to do {k}-shot. Has {len(grade_df)}.")
                    sampled = grade_df.index.tolist()
                else:
                    sampled = rng.choice(grade_df.index.tolist(), size=k, replace=False).tolist()
                support_indices.extend(sampled)
            
            support_df = df_train.loc[support_indices]
            query_df = df_train.drop(support_indices)
            
            support_file = os.path.join(out_dir, f"{mod_lower}_{k}shot_s{seed}_support.txt")
            query_file = os.path.join(out_dir, f"{mod_lower}_{k}shot_s{seed}_query.txt")
            
            # Save absolute paths
            support_df['path'].to_csv(support_file, index=False, header=False)
            query_df['path'].to_csv(query_file, index=False, header=False)
            
            print(f"Generated {mod_lower} {k}-shot seed {seed}: {len(support_df)} support, {len(query_df)} query")
        
        # Base-to-novel: support has grades 0-3, novel is grade 4
        # Just creating one support/query file per seed for the base-to-novel setting
        # We sample e.g. 16 shots per base class
        k_b2n = 16
        b2n_support = []
        for grade in [0, 1, 2, 3]:
            grade_df = df_train[df_train['grade'] == grade]
            if len(grade_df) >= k_b2n:
                b2n_support.extend(rng.choice(grade_df.index.tolist(), size=k_b2n, replace=False).tolist())
            else:
                b2n_support.extend(grade_df.index.tolist())
                
        b2n_support_df = df_train.loc[b2n_support]
        # query is remaining base + all novel
        b2n_query_df = df_train.drop(b2n_support)
        
        support_file = os.path.join(out_dir, f"{mod_lower}_b2n_s{seed}_support.txt")
        query_file = os.path.join(out_dir, f"{mod_lower}_b2n_s{seed}_query.txt")
        b2n_support_df['path'].to_csv(support_file, index=False, header=False)
        b2n_query_df['path'].to_csv(query_file, index=False, header=False)

if __name__ == "__main__":
    main()
