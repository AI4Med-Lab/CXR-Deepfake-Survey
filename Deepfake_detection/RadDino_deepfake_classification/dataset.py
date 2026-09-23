"""
dataset.py -- reads metadata.csv produced by prepare_data.py and serves
(pixel_values, label) pairs ready for the RAD-DINO image processor.
"""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CXRDeepfakeDataset(Dataset):
    def __init__(self, csv_path: str, split: str, processor, augment=None):
        """
        csv_path: path to metadata.csv (columns: path, label, patient_id, method, split)
        split: one of "train", "val", "test"
        processor: a transformers AutoImageProcessor for microsoft/rad-dino
        augment: optional torchvision transform applied to the PIL image
                 BEFORE the RAD-DINO processor runs (resize/normalize happens in the processor)
        """
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={split!r} in {csv_path}")
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.augment is not None:
            image = self.augment(image)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        label = torch.tensor(row['label'], dtype=torch.float32)
        return pixel_values, label