import os
import json
import numpy as np
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    all_data = []
    
    for config_name in os.listdir(args.results_dir):
        config_dir = os.path.join(args.results_dir, config_name)
        if not os.path.isdir(config_dir):
            continue
            
        # e.g., CFP_4shot_s0
        for run_name in os.listdir(config_dir):
            run_dir = os.path.join(config_dir, run_name)
            stats_file = os.path.join(run_dir, "test_stats.json")
            
            if os.path.exists(stats_file):
                parts = run_name.split('_')
                modality = parts[0]
                k_shot = int(parts[1].replace('shot', ''))
                seed = int(parts[2].replace('s', ''))
                
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    
                row = {
                    'Config': config_name,
                    'Modality': modality,
                    'k': k_shot,
                    'Seed': seed,
                    **stats
                }
                all_data.append(row)
                
    if not all_data:
        print("No results found.")
        return
        
    df = pd.DataFrame(all_data)
    
    # Aggregate mean and std
    metrics = ['qwk', 'mae', 'f1', 'auc', 'acc', 'los']
    df_agg = df.groupby(['Config', 'Modality', 'k'])[metrics].agg(['mean', 'std']).reset_index()
    
    # Format output
    for m in metrics:
        df_agg[m + '_str'] = df_agg.apply(lambda r: f"{r[(m, 'mean')]:.4f} ± {r[(m, 'std')]:.4f}", axis=1)
        
    final_cols = ['Config', 'Modality', 'k'] + [m + '_str' for m in metrics]
    df_final = df_agg[final_cols]
    
    df_final.columns = ['Config', 'Modality', 'k', 'QWK', 'MAE', 'F1', 'AUC', 'ACC', 'LOS']
    print(df_final.to_markdown(index=False))
    
    out_path = os.path.join(args.results_dir, "aggregated_results.csv")
    df_final.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
