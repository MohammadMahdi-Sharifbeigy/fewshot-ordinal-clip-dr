import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import json

class MMRDRDataset(Dataset):
    def __init__(self, manifest_csv, split_file=None, split_type=None, transform=None, paths=None):
        """
        manifest_csv: path to the modality manifest.csv
        split_file: path to a text file containing paths (e.g. from build_splits.py), OR
        split_type: "train" or "test" to use the original dataset splits, OR
        paths: an explicit list of image paths to keep.
        transform: torchvision transforms
        """
        super().__init__()
        self.transform = transform
        self.df = pd.read_csv(manifest_csv)

        if paths is not None:
            self.df = self.df[self.df['path'].isin(set(paths))].reset_index(drop=True)
        elif split_file is not None:
            with open(split_file, 'r') as f:
                paths = [line.strip() for line in f if line.strip()]
            self.df = self.df[self.df['path'].isin(paths)].reset_index(drop=True)
        elif split_type is not None:
            self.df = self.df[self.df['split'] == split_type].reset_index(drop=True)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        grade = int(row['grade'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # return a black image on failure to not crash
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform is not None:
            image = self.transform(image)
            
        lesion_str = str(row['lesion']) if 'lesion' in row and not pd.isna(row['lesion']) else ""
        lesions = []
        if lesion_str.startswith('['):
            try:
                lesions = json.loads(lesion_str)
            except:
                pass
                
        lesion_tensor = torch.tensor(lesions, dtype=torch.float32) if lesions else torch.empty(0)
        
        return image, grade, lesion_tensor, img_path
