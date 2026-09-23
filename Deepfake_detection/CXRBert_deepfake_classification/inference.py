"""
inference.py -- run a trained CXRBertClassifier checkpoint over a data split
(default "test") and write per-example predictions + summary metrics.

Usage:
    python inference.py \
        --csv /path/to/train_val_split.csv \
        --split test \
        --checkpoint /path/to/Results/best_model_finetune.pt \
        --output_csv /path/to/Results/test_predictions.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import CXRReportDataset
from model import CXRBertClassifier



@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_logits = []
    for input_ids, attention_mask, _labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = model(input_ids, attention_mask)
        all_logits.append(logits.cpu())
    logits = torch.cat(all_logits).numpy()
    probs = 1 / (1 + np.exp(-logits))
    return probs


def main():

    args = {
        "csv": "",
        "split": "test",
        "backbone": "microsoft/BiomedVLP-CXR-BERT-specialized",
        "checkpoint": "",
        "output_csv": "",
        "batch_size": 32,
        "num_workers": 4,
        "max_text_length": 256,
        "threshold": 0.5,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args['backbone'], trust_remote_code=True)
    dataset = CXRReportDataset(args['csv'], args['split'], tokenizer, max_length=args['max_text_length'])
    loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers'],
    )

    model = CXRBertClassifier(args['backbone'], num_unfrozen_layers=0).to(device)
    state_dict = torch.load(args['checkpoint'], map_location=device)
    model.load_state_dict(state_dict)

    probs = run_inference(model, loader, device)
    preds = (probs >= args['threshold']).astype(int)
    labels = dataset.df["label"].to_numpy()

    out_df = dataset.df.copy()
    out_df["prediction_prob"] = probs
    out_df["prediction"] = preds
    os.makedirs(os.path.dirname(args['output_csv']) or ".", exist_ok=True)
    out_df.to_csv(args['output_csv'], index=False)
    print(f"Wrote per-example predictions to {args['output_csv']}")



if __name__ == "__main__":
    main()