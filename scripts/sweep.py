import os
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    k_shots = [1, 2, 4, 8, 16]
    seeds = [0, 1, 2, 3, 4]
    
    for k in k_shots:
        for s in seeds:
            print(f"Running config={args.config} k={k} seed={s}")
            cmd = f"python scripts/train.py --config {args.config} data.k_shot={k} data.seed={s}"
            subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main()
