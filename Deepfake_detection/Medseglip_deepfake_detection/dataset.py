"""
dataset.py -- reads metadata.csv produced by prepare_data.py and serves
(pixel_values, input_ids, attention_mask, label) tuples ready for a
MedSigLIP (image + text) classifier.

Expected CSV columns: image_path, label, report, patient_id, method, split
"""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CXRDeepfakeDataset(Dataset):
    def __init__(self, csv_path: str, split: str, processor, augment=None, max_length: int = 64):
        """
        csv_path: path to metadata.csv (columns: image_path, label, report, patient_id, method, split)
        split: one of "train", "val", "test"
        processor: a transformers AutoProcessor for the MedSigLIP backbone (handles both
                   image preprocessing and text tokenization)
        augment: optional torchvision transform applied to the PIL image
                 BEFORE the processor runs (resize/normalize happens in the processor)
        max_length: max token length for the text report (SigLIP-style models generally
                    pad to a fixed length rather than using dynamic padding)
        """
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={split!r} in {csv_path}")
        if "report" not in self.df.columns:
            raise ValueError(f"Expected a 'report' column in {csv_path} for text encoding")
        self.processor = processor
        self.augment = augment
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.augment is not None:
            image = self.augment(image)

        report = row["report"]
        report = "" if pd.isna(report) else str(report)

        encoded = self.processor(
            images=image,
            text=report,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        pixel_values = encoded["pixel_values"][0]
        input_ids = encoded["input_ids"][0]
        # Some SigLIP-family processors don't return an attention_mask (they always
        # pad to max_length and rely on the pad token), so fall back to all-ones.
        if "attention_mask" in encoded:
            attention_mask = encoded["attention_mask"][0]
        else:
            attention_mask = torch.ones_like(input_ids)

        label = torch.tensor(row["label"], dtype=torch.float32)
        return pixel_values, input_ids, attention_mask, label