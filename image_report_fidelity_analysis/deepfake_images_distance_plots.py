"""
Standalone script: nearest-neighbor-distance-to-real box/violin plot only.
Does not recompute FID/KID/Precision-Recall or load BiomedCLIP -- just
RAD-DINO features, so it's much faster than re-running the full suite.

Saves into the SAME output folder as cxr_similarity_suite.py, so it lands
alongside metrics_summary.csv and the other plots.

Edit the CONFIG section below directly (no command-line arguments).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModel

# =============================================================================
# CONFIG -- must match cxr_similarity_suite.py so results are comparable
# =============================================================================
N_IMAGES = 1000
BATCH_SIZE = 32
SEED = 45
OUTPUT_DIR = Path("./cxr_similarity_outputs")  # same folder as the main suite script

REAL_CSV = {
    "path": "/home/2023eeb1196/Deepfake_in_healthcare/Deepfake_Reports/generated_prompts_labels_2.csv",
    "column": "ImagePath",
    "base_dir": None,
}

# Generated images: one entry per method. Add/remove entries freely -- no
# code changes needed elsewhere. Each entry can use a different column name
# and base_dir if its CSV is laid out differently.
FAKE_CSVS = [
    {"name": "CXR-IRGen", "path": "/home/2023eeb1196/Deepfake_in_healthcare/Baselines/CXR-IRGen/output/generation_metadata.csv",
     "column": "generated_image_path", "base_dir": None},
    {"name": "ProgEmu", "path": "/home/2023eeb1196/Deepfake_in_healthcare/Baselines/ProgEmu/outputs/inference_samples.csv",
     "column": "image_path", "base_dir": None},
    {"name": "Qwen", "path": "/home/2023eeb1196/Deepfake_in_healthcare/Baselines/Qwen-Image-Edit-2511/outputs/generation_summary.csv",
     "column": "GeneratedImagePath", "base_dir": None},
    # {"name": "NewMethod", "path": "/path/to/new_method.csv",
    #  "column": "generated_image_path", "base_dir": None},
]

RAD_DINO_MODEL_ID = "microsoft/rad-dino"

METHOD_COLORS = {
    "Real": "#7f7f7f",
    "CXR-IRGen": "#2ca02c",
    "ProgEmu": "#1f77b4",
    "Qwen": "#d62728",
}

# =============================================================================


def load_paths_from_csv(csv_path, column, base_dir, n, seed):
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}. Columns: {list(df.columns)}")
    paths = df[column].dropna().astype(str).tolist()
    if not paths:
        raise ValueError(f"No paths found in column '{column}' of {csv_path}")
    if base_dir:
        paths = [str(Path(base_dir) / p) for p in paths]
    if len(paths) < n:
        print(f"Warning: {csv_path} has only {len(paths)} usable rows, fewer than requested {n}.")
    rng = np.random.RandomState(seed)
    if len(paths) > n:
        idx = rng.choice(len(paths), size=n, replace=False)
        paths = [paths[i] for i in idx]
    return [Path(p) for p in paths]


@torch.no_grad()
def extract_rad_dino_features(processor, model, paths, device):
    feats = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="RAD-DINO features"):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_SIZE]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        out = model(**inputs)
        feats.append(out.pooler_output.cpu().numpy())
    return np.concatenate(feats, axis=0)


def nearest_neighbor_distances(real_feats, query_feats, exclude_self=False):
    """Distance from each query point to its nearest point in real_feats.
    exclude_self=True is for the Real-vs-Real baseline (query_feats IS
    real_feats), where each point's nearest neighbor must be a DIFFERENT
    real image, not itself (distance 0)."""
    dists = cdist(query_feats, real_feats)
    if exclude_self:
        np.fill_diagonal(dists, np.inf)
    return dists.min(axis=1)


def plot_nn_distance_boxplot(features_by_source, output_path):
    real_feats = features_by_source["Real"]
    data = {"Real\n(self-NN baseline)": nearest_neighbor_distances(real_feats, real_feats, exclude_self=True)}
    for source, feats in features_by_source.items():
        if source != "Real":
            data[source] = nearest_neighbor_distances(real_feats, feats)

    labels = list(data.keys())
    box_data = list(data.values())
    colors = [METHOD_COLORS.get(l.split("\n")[0], "#999999") for l in labels]
    positions = range(len(data))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    parts = ax.violinplot(box_data, positions=positions, showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    bp = ax.boxplot(box_data, positions=positions, widths=0.15, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2),
                     flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Distance to nearest REAL image\n(RAD-DINO feature space, Euclidean)")
    ax.set_title("Nearest-Neighbor Distance to Real\n(lower = closer to real; not projection-distorted, unlike t-SNE)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading RAD-DINO ({RAD_DINO_MODEL_ID}) ...")
    processor = AutoImageProcessor.from_pretrained(RAD_DINO_MODEL_ID)
    model = AutoModel.from_pretrained(RAD_DINO_MODEL_ID).to(device).eval()

    sources = {"Real": REAL_CSV}
    for cfg in FAKE_CSVS:
        sources[cfg["name"]] = cfg

    features_by_source = {}
    for name, cfg in sources.items():
        paths = load_paths_from_csv(cfg["path"], cfg["column"], cfg["base_dir"], N_IMAGES, SEED)
        print(f"{name}: {len(paths)} images from {cfg['path']}")
        features_by_source[name] = extract_rad_dino_features(processor, model, paths, device)

    output_path = OUTPUT_DIR / "nn_distance_boxplot.png"
    plot_nn_distance_boxplot(features_by_source, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()