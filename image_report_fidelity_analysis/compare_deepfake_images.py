"""
Comparison suite for measuring how far each generation method's images are
from real CXRs, using TWO different feature extractors and THREE different
metrics, plus three plots that go beyond a single bar/radar chart.

Feature extractors (deliberately different training paradigms, to
cross-validate that any gap found isn't an artifact of one model's biases):
  - RAD-DINO       : self-supervised ViT, trained only on chest X-rays
  - BiomedCLIP      : CLIP-style, trained on biomedical image-text pairs
                      across many modalities (not CXR-specific)

Metrics:
  - FID (Frechet distance)  : standard, but assumes Gaussian features
  - KID (polynomial-kernel MMD): unbiased, no Gaussian assumption, generally
                                 more reliable at moderate sample sizes
  - Precision & Recall (Kynkaanniemi et al. 2019): splits "how different"
    into FIDELITY (do fake images look realistic?) and DIVERSITY (do fakes
    cover the full range of real variation, or collapse to a narrow mode?)
    -- something a single FID/KID number cannot distinguish.

Edit the CONFIG section below directly (no command-line arguments).
FAKE_CSVS is a list -- append a new dict to add another generation method.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import linalg
from scipy.spatial.distance import cdist
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoImageProcessor, AutoModel
from open_clip import create_model_from_pretrained

# =============================================================================
# CONFIG -- edit these directly
# =============================================================================
N_IMAGES = 1000
BATCH_SIZE = 32
SEED = 42
KNN_K = 3           # k for Precision/Recall manifold estimation
OUTPUT_DIR = Path("./cxr_similarity_outputs")

REAL_CSV = {
    "path": "",
    "column": "ImagePath",
    "base_dir": None,
}

# Generated images: one entry per method. Add/remove entries freely -- no
# code changes needed elsewhere. Each entry can use a different column name
# and base_dir if its CSV is laid out differently.
FAKE_CSVS = [
    {"name": "CXR-IRGen", "path": "",
     "column": "generated_image_path", "base_dir": None},
    {"name": "ProgEmu", "path": "",
     "column": "image_path", "base_dir": None},
    {"name": "Qwen", "path": "",
     "column": "GeneratedImagePath", "base_dir": None},
]

FEATURE_EXTRACTORS = [
    {"name": "RAD-DINO", "model_id": "microsoft/rad-dino", "kind": "dinov2"},
    {"name": "BiomedCLIP", "model_id": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
     "kind": "biomedclip"},
]
PRIMARY_EXTRACTOR = "RAD-DINO"  # used for Precision/Recall and the UMAP plot

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


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------
def build_extractor(cfg, device):
    if cfg["kind"] == "dinov2":
        processor = AutoImageProcessor.from_pretrained(cfg["model_id"])
        model = AutoModel.from_pretrained(cfg["model_id"]).to(device).eval()

        @torch.no_grad()
        def extract(paths):
            feats = []
            for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=f"{cfg['name']} features"):
                batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_SIZE]]
                inputs = processor(images=batch, return_tensors="pt").to(device)
                out = model(**inputs)
                feats.append(out.pooler_output.cpu().numpy())
            return np.concatenate(feats, axis=0)
        return extract

    elif cfg["kind"] == "biomedclip":
        model, preprocess = create_model_from_pretrained(f"hf-hub:{cfg['model_id']}")
        model = model.to(device).eval()

        @torch.no_grad()
        def extract(paths):
            feats = []
            for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=f"{cfg['name']} features"):
                batch = torch.stack(
                    [preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + BATCH_SIZE]]
                ).to(device)
                image_features = model.encode_image(batch)
                feats.append(image_features.cpu().numpy())
            return np.concatenate(feats, axis=0)
        return extract

    raise ValueError(f"Unknown extractor kind: {cfg['kind']}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def polynomial_kernel(X, Y, degree=3, coef0=1.0):
    d = X.shape[1]
    return (X.dot(Y.T) / d + coef0) ** degree


def kernel_inception_distance(X, Y, degree=3, coef0=1.0):
    """Unbiased MMD^2 estimator with a degree-3 polynomial kernel (standard
    KID formulation, Binkowski et al. 2018)."""
    m, n = X.shape[0], Y.shape[0]
    Kxx = polynomial_kernel(X, X, degree, coef0)
    Kyy = polynomial_kernel(Y, Y, degree, coef0)
    Kxy = polynomial_kernel(X, Y, degree, coef0)

    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sum_xy = Kxy.sum() / (m * n)
    return float(sum_xx + sum_yy - 2 * sum_xy)


def knn_radii(features, k):
    dists = cdist(features, features)
    np.fill_diagonal(dists, np.inf)
    return np.sort(dists, axis=1)[:, k - 1]


def precision_recall(real_feats, fake_feats, k=3):
    """Kynkaanniemi et al. 2019 manifold-based Precision/Recall.
    Precision: fraction of FAKE samples that fall inside the real manifold
               (i.e. within some real point's k-NN radius) -- FIDELITY.
    Recall:    fraction of REAL samples that fall inside the fake manifold
               -- DIVERSITY (does the fake set cover the real variation?)."""
    real_radii = knn_radii(real_feats, k)
    fake_radii = knn_radii(fake_feats, k)

    d_fake_to_real = cdist(fake_feats, real_feats)
    precision = float(np.mean(np.any(d_fake_to_real <= real_radii[None, :], axis=1)))

    d_real_to_fake = cdist(real_feats, fake_feats)
    recall = float(np.mean(np.any(d_real_to_fake <= fake_radii[None, :], axis=1)))
    return precision, recall


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_umap_manifold(features_by_source, output_path):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=SEED)
        label = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=SEED, init="pca")
        label = "t-SNE"

    sources = list(features_by_source.keys())
    all_feats = np.concatenate([features_by_source[s] for s in sources], axis=0)
    embedding = reducer.fit_transform(all_feats)

    fig, ax = plt.subplots(figsize=(8, 7))
    start = 0
    for source in sources:
        n = len(features_by_source[source])
        pts = embedding[start:start + n]
        start += n
        color = METHOD_COLORS.get(source, None)
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.5, color=color, label=source)
        if n > 10:
            sns.kdeplot(x=pts[:, 0], y=pts[:, 1], ax=ax, color=color,
                        levels=3, linewidths=1.2, alpha=0.6)

    ax.set_title(f"{label} Projection of RAD-DINO Feature Space\n(density contours show each source's manifold)")
    ax.set_xlabel(f"{label} 1")
    ax.set_ylabel(f"{label} 2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(pr_results, output_path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, (precision, recall) in pr_results.items():
        color = METHOD_COLORS.get(name, None)
        ax.scatter(recall, precision, s=180, color=color, edgecolor="black", zorder=5)
        ax.annotate(name, (recall, precision), textcoords="offset points",
                    xytext=(8, 6), fontsize=10)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall (diversity: coverage of real variation)")
    ax.set_ylabel("Precision (fidelity: realism of generated images)")
    ax.set_title("Fidelity vs. Diversity per Generation Method")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bump_chart(metrics_df, output_path):
    """One line per method, showing its RANK (1 = best/closest to real) on
    each metric. All metrics are oriented so LOWER rank number = closer to
    real, for direct visual comparison. Consistent low ranks across every
    metric = a robust finding, not an artifact of one measurement choice."""
    ranked = metrics_df.copy()
    for col in ranked.columns:
        if col.startswith("Precision") or col.startswith("Recall"):
            ranked[col] = ranked[col].rank(ascending=False)  # higher P/R = better = rank 1
        else:
            ranked[col] = ranked[col].rank(ascending=True)   # lower FID/KID = better = rank 1

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ranked.columns))
    for method in ranked.index:
        color = METHOD_COLORS.get(method, None)
        ax.plot(x, ranked.loc[method], marker="o", linewidth=2, markersize=8,
                color=color, label=method)
        ax.text(x[-1] + 0.08, ranked.loc[method, ranked.columns[-1]], method,
                va="center", fontsize=9, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(ranked.columns, rotation=20, ha="right")
    ax.invert_yaxis()  # rank 1 (best) at the top
    ax.set_ylabel("Rank (1 = closest to real)")
    ax.set_title("Method Ranking Across All Metrics\n(agreement across rows = robust finding)")
    ax.set_yticks(range(1, len(ranked.index) + 1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image paths once (shared across all feature extractors)
    sources = {"Real": REAL_CSV}
    for cfg in FAKE_CSVS:
        sources[cfg["name"]] = cfg

    paths_by_source = {}
    for name, cfg in sources.items():
        paths = load_paths_from_csv(cfg["path"], cfg["column"], cfg["base_dir"], N_IMAGES, SEED)
        print(f"{name}: {len(paths)} images from {cfg['path']}")
        paths_by_source[name] = paths

    # Extract features from every extractor, for every source
    features = {}  # features[extractor_name][source_name] = np.ndarray
    for ext_cfg in FEATURE_EXTRACTORS:
        print(f"\nLoading extractor: {ext_cfg['name']} ({ext_cfg['model_id']}) ...")
        extract = build_extractor(ext_cfg, device)
        features[ext_cfg["name"]] = {}
        for name, paths in paths_by_source.items():
            features[ext_cfg["name"]][name] = extract(paths)

    # FID + KID for every (extractor, fake method) pair
    rows = {cfg["name"]: {} for cfg in FAKE_CSVS}
    for ext_name, feats_by_source in features.items():
        real_feats = feats_by_source["Real"]
        mu_real, sigma_real = real_feats.mean(0), np.cov(real_feats, rowvar=False)
        for cfg in FAKE_CSVS:
            fake_feats = feats_by_source[cfg["name"]]
            mu_fake, sigma_fake = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
            fid = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
            kid = kernel_inception_distance(real_feats, fake_feats)
            rows[cfg["name"]][f"FID ({ext_name})"] = fid
            rows[cfg["name"]][f"KID ({ext_name})"] = kid

    # Precision/Recall using the primary extractor only
    primary_feats = features[PRIMARY_EXTRACTOR]
    pr_results = {}
    for cfg in FAKE_CSVS:
        precision, recall = precision_recall(primary_feats["Real"], primary_feats[cfg["name"]], k=KNN_K)
        pr_results[cfg["name"]] = (precision, recall)
        rows[cfg["name"]]["Precision"] = precision
        rows[cfg["name"]]["Recall"] = recall

    metrics_df = pd.DataFrame(rows).T
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv")
    print("\n=== Metrics summary ===")
    print(metrics_df.round(3).to_string())

    # Plots
    print("\nGenerating plots ...")
    plot_umap_manifold(primary_feats, OUTPUT_DIR / "manifold_projection.png")
    plot_precision_recall(pr_results, OUTPUT_DIR / "precision_recall.png")
    plot_bump_chart(metrics_df, OUTPUT_DIR / "rank_agreement.png")
    print(f"Saved metrics_summary.csv and 3 plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()