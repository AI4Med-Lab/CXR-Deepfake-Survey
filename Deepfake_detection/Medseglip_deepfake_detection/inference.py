import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from dataset import CXRDeepfakeDataset
from model import MedSiglipDeepfakeClassifier


@torch.no_grad()
def main():
    args = {
        "csv": "",
        "split": "test",
        "backbone": "google/medsiglip-448",
        "batch_size": 32,
        "num_workers": 4,
        "max_text_length": 64,
        "model_path": "",
        "output_csv": "",
        "prediction_threshold": 0.5,
    }

    metadata = pd.read_csv(args["csv"])
    if "split" not in metadata.columns:
        raise ValueError(f"Expected a 'split' column in {args['csv']}")

    test_metadata = metadata[metadata["split"] == args["split"]].reset_index(drop=True)
    if test_metadata.empty:
        raise ValueError(f"No rows found for split={args['split']!r} in {args['csv']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(
        args["backbone"],
        token="",
    )
    dataset = CXRDeepfakeDataset(
        args["csv"],
        args["split"],
        processor,
        max_length=args["max_text_length"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=device == "cuda",
    )

    model = MedSiglipDeepfakeClassifier(args["backbone"]).to(device)
    checkpoint = torch.load(args["model_path"], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    predictions = []
    for pixel_values, input_ids, attention_mask, _ in loader:
        logits = model(
            pixel_values.to(device),
            input_ids.to(device),
            attention_mask.to(device),
        )
        probabilities = torch.sigmoid(logits)
        predictions.extend(
            (probabilities >= args["prediction_threshold"]).to(torch.int64).cpu().tolist()
        )

    if len(predictions) != len(test_metadata):
        raise RuntimeError(
            f"Generated {len(predictions)} predictions for {len(test_metadata)} test rows"
        )

    results = test_metadata.copy()
    results["prediction"] = predictions
    output_dir = os.path.dirname(args["output_csv"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results.to_csv(args["output_csv"], index=False)
    print(f"Saved {len(results)} test predictions to {args['output_csv']}")


if __name__ == "__main__":
    main()