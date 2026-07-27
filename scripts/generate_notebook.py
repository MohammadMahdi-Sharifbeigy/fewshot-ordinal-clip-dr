import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Markdown
cells.append(nbf.v4.new_markdown_cell("# Interactive Few-Shot Ordinal-Aware CLIP for DR Grading\n\nThis notebook demonstrates how to load the MMRDR dataset, visualize the data, construct the L2RCLIP + CLIP-DR framework, and run the training loop entirely within the notebook."))

# Code
cells.append(nbf.v4.new_code_cell("""import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# Ensure we are in the Workspace root directory
while not os.path.exists('configs') and os.path.basename(os.getcwd()) != 'MMRDR':
    os.chdir('..')

# Make sure src can be imported
sys.path.append(os.getcwd())

print("Current Working Directory:", os.getcwd())
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load Configuration\nWe will use the CFP config for a 4-shot experiment."))

cells.append(nbf.v4.new_code_cell("""# Load the base config and CFP-specific overrides
base_cfg = OmegaConf.load('configs/default.yaml')
cfp_cfg = OmegaConf.load('configs/mrdr_cfp.yaml')
cfg = OmegaConf.merge(base_cfg, cfp_cfg)

# Set specific few-shot parameters for this interactive run
cfg.data.k_shot = 4
cfg.data.seed = 1 # Must be 1-5 as per the generated splits
cfg.train.epochs = 3 # Keep it short for demonstration

print("Configuration:")
print(OmegaConf.to_yaml(cfg))
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Initialize DataModule and Visualize Data\nLet's load the PyTorch Lightning DataModule. We assume `prepare_mmrdr.py` and `build_splits.py` have already generated the manifests and splits."))

cells.append(nbf.v4.new_code_cell("""from src.data.datamodule import FewShotDataModule

pl.seed_everything(cfg.data.seed)
datamodule = FewShotDataModule(cfg)
datamodule.setup()

print(f"Support set size (Train): {len(datamodule.support_dataset)}")
print(f"Query set size (Val): {len(datamodule.query_dataset)}")
print(f"Test set size (Test): {len(datamodule.test_dataset)}")
"""))

cells.append(nbf.v4.new_code_cell("""# Visualize a few support images
support_loader = datamodule.train_dataloader()
images, targets, lesions, paths = next(iter(support_loader))

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    # Un-normalize image for visualization
    img = images[i].permute(1, 2, 0).numpy()
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    axes[i].imshow(img)
    axes[i].set_title(f"Rank/Grade: {targets[i].item()}")
    axes[i].axis('off')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Initialize the Model\nWe build the `FewShotOrdinalRunner` which contains CLIP, LoRA, the RankFormer Prompt Learner, and the SMS calibration module."))

cells.append(nbf.v4.new_code_cell("""from src.models.runner import FewShotOrdinalRunner

model = FewShotOrdinalRunner(cfg)
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Training Loop\nWe use PyTorch Lightning's Trainer to handle the loop. We will enable the progress bar to see the training happen step-by-step."))

cells.append(nbf.v4.new_code_cell("""# Initialize the Trainer
trainer = pl.Trainer(
    max_epochs=cfg.train.epochs,
    accelerator="auto",
    devices=1,
    precision="16-mixed",
    log_every_n_steps=1,
    enable_checkpointing=False # Disable saving weights for this demo
)

# Start training!
trainer.fit(model, datamodule=datamodule)
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Evaluation and Analysis\nFinally, we evaluate on the unseen test set and plot a confusion matrix."))

cells.append(nbf.v4.new_code_cell("""# Run testing
trainer.test(model, datamodule=datamodule)

# Fetch latest metrics logged by the model
print("\\nFinal Metrics on Test Set:")
for k, v in model.latest_metrics.items():
    print(f"{k.upper()}: {v:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""# Plot Confusion Matrix
# We can access the stored predictions and targets from the validation/test step
# (We'll re-run a quick inference loop over a subset of test data for the plot)
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

all_preds = []
all_targets = []

with torch.no_grad():
    for images, targets, _, _ in datamodule.test_dataloader():
        images = images.to(model.device)
        image_features, text_features = model(images)
        
        logit_scale = model.clip_model.logit_scale.exp()
        sim = logit_scale * image_features @ text_features.T
        
        if model.sms is not None:
            sim = model.sms(sim)
            
        preds = sim.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
        
        if len(all_preds) > 500: # Stop early for fast plotting
            break

cm = confusion_matrix(all_targets, all_preds, labels=range(cfg.model.prompt_learner.num_ranks))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(cfg.model.prompt_learner.num_ranks))
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title("Test Set Confusion Matrix")
plt.show()
"""))

nb['cells'] = cells

with open('notebooks/MMRDR_FewShot_DR.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Interactive Usage Notebook generated successfully.")
