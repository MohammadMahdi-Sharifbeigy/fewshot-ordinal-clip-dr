import os
import pytest
from omegaconf import OmegaConf
from src.data.datamodule import FewShotDataModule

def test_sampler():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_dir = os.path.join(workspace_dir, "data", "MMRDR", "splits")
    manifest_path = os.path.join(workspace_dir, "data", "MMRDR", "CFP_manifest.csv")
    
    if not os.path.exists(splits_dir) or not os.path.exists(manifest_path):
        pytest.skip("Data not prepared yet.")
        
    cfg = OmegaConf.create({
        "data": {
            "modality": "CFP",
            "k_shot": 2,
            "seed": 1,
            "batch_query": 32,
            "image_size": 224,
            "manifest": manifest_path,
            "splits_dir": splits_dir,
            "num_workers": 0
        }
    })
    
    dm = FewShotDataModule(cfg)
    dm.setup()
    
    # The support dataloader should have all 10 images (5 ranks * 2 shots)
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    images, targets, lesions, paths = batch
    
    assert images.shape == (10, 3, 224, 224)
    assert targets.shape == (10,)
    
    # Check that query has no support paths
    support_paths = set(paths)
    query_dl = dm.val_dataloader()
    query_batch = next(iter(query_dl))
    query_paths = set(query_batch[3])
    
    assert len(support_paths.intersection(query_paths)) == 0
