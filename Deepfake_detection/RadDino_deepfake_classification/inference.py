"""
Run inference on images listed in a CSV file and save predictions.
"""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from dataset import CXRDeepfakeDataset
from model import RadDinoDeepfakeClassifier


def main():
    args = {
        "checkpoint": (
            ""
        ),
        "backbone": "microsoft/rad-dino",
        "image_csv_path": (
            ""
        ),
        "results_dir": (
            ""
        ),
        "output_filename": "predictions_final_bestchkpt.csv",
        "threshold": 0.5,
        "batch_size": 16,
        "num_workers": 4,
    }

    os.makedirs(args["results_dir"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(args["backbone"])

    test_dataset = CXRDeepfakeDataset(
        args["image_csv_path"],
        "test",
        processor,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=(device == "cuda"),
    )

    model = RadDinoDeepfakeClassifier(args["backbone"])
    checkpoint = torch.load(args["checkpoint"], map_location=device)

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    probabilities = []
    logits = []

    with torch.inference_mode():
        for pixel_values, _ in test_loader:
            pixel_values = pixel_values.to(device)
            batch_logits = model(pixel_values).view(-1)
            batch_probabilities = torch.sigmoid(batch_logits)

            logits.extend(batch_logits.cpu().tolist())
            probabilities.extend(batch_probabilities.cpu().tolist())

    predictions = [
        int(probability >= args["threshold"])
        for probability in probabilities
    ]

    results = test_dataset.df.copy()
    results["prediction_logit"] = logits
    results["prediction_probability"] = probabilities
    results["prediction"] = predictions

    output_path = os.path.join(
        args["results_dir"],
        args["output_filename"],
    )
    results.to_csv(output_path, index=False)

    print(f"Saved {len(results)} predictions to: {output_path}")


if __name__ == "__main__":
    main()