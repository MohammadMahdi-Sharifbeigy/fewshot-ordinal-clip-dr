import os
import pytest
import pandas as pd

def test_splits_properties():
    # Only test if files exist
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_dir = os.path.join(workspace_dir, "data", "MMRDR", "splits")
    manifest_path = os.path.join(workspace_dir, "data", "MMRDR", "CFP_manifest.csv")
    
    if not os.path.exists(splits_dir) or not os.path.exists(manifest_path):
        pytest.skip("Data not prepared yet.")
        
    df = pd.read_csv(manifest_path)
    
    k = 4
    seed = 1
    support_file = os.path.join(splits_dir, f"cfp_{k}shot_s{seed}_support.txt")
    query_file = os.path.join(splits_dir, f"cfp_{k}shot_s{seed}_query.txt")
    
    assert os.path.exists(support_file)
    assert os.path.exists(query_file)
    
    with open(support_file, 'r') as f:
        support_paths = [l.strip() for l in f if l.strip()]
        
    with open(query_file, 'r') as f:
        query_paths = [l.strip() for l in f if l.strip()]
        
    # Check disjoint
    support_set = set(support_paths)
    query_set = set(query_paths)
    assert len(support_set.intersection(query_set)) == 0, "Overlap between support and query!"
    
    # Check k per rank in support
    support_df = df[df['path'].isin(support_paths)]
    counts = support_df['grade'].value_counts()
    for grade in range(5):
        assert counts.get(grade, 0) == k, f"Expected {k} samples for grade {grade}, got {counts.get(grade, 0)}"
