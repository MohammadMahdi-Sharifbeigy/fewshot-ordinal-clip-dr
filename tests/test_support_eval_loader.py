import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data.datamodule import FewShotDataModule


def _tiny_k_shot_setup(tmp_path):
    rows, paths = ["path,modality,grade,lesion,lr,split"], []
    for g in range(5):
        p = tmp_path / f"img{g}.png"
        Image.new("RGB", (64, 48), (40 * g, 10, 200 - 30 * g)).save(p)
        rows.append(f"{p},CFP,{g},,0,train")
        paths.append(str(p))
    manifest = tmp_path / "m.csv"
    manifest.write_text("\n".join(rows) + "\n")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "cfp_1shot_s1_support.txt").write_text("\n".join(paths) + "\n")
    (splits / "cfp_1shot_s1_query.txt").write_text("\n".join(paths) + "\n")
    cfg = OmegaConf.create({"data": {"modality": "CFP", "k_shot": 1, "seed": 1, "batch_query": 32,
                                     "image_size": 32, "manifest": str(manifest),
                                     "splits_dir": str(splits), "num_workers": 0}})
    dm = FewShotDataModule(cfg)
    dm.setup()
    return dm


def test_support_eval_loader_is_deterministic_and_unshuffled(tmp_path):
    # CLAP's zero-shot λ must come from clean support images: no augmentation, fixed order
    dm = _tiny_k_shot_setup(tmp_path)
    a = [b for b in dm.support_eval_dataloader()]
    b = [b for b in dm.support_eval_dataloader()]
    assert torch.equal(a[0][0], b[0][0])
    assert a[0][1].tolist() == [0, 1, 2, 3, 4]
