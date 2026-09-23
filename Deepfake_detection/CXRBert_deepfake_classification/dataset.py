"""
dataset.py -- reads metadata.csv (with a 'report' text column and a 'label'
column) and serves (input_ids, attention_mask, label) tuples for CXR-BERT.

Expected CSV columns: report, label, patient_id, method, split
(image_path may also be present but is unused here -- this is text-only.)
"""
import pandas as pd
import torch
from torch.utils.data import Dataset


class CXRReportDataset(Dataset):
    def __init__(self, csv_path: str, split: str, tokenizer, max_length: int = 256):
        """
        csv_path: path to metadata.csv
        split: one of "train", "val", "test"
        tokenizer: a transformers AutoTokenizer for the CXR-BERT backbone
        max_length: max token length for the report text
        """
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={split!r} in {csv_path}")
        if "report" not in self.df.columns:
            raise ValueError(f"Expected a 'report' column in {csv_path}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        report = row["report"]
        report = "" if pd.isna(report) else str(report)

        encoded = self.tokenizer(
            report,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        label = torch.tensor(row["label"], dtype=torch.float32)
        return input_ids, attention_mask, label