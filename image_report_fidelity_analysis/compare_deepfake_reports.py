"""
Deepfake evaluation suite for GENERATED RADIOLOGY REPORTS (text), paralleling
the image-side suite. Edit the CONFIG section below directly.

Four questions this addresses, and why each metric was chosen:

Q1 -- Distributional realism (do generated reports read like real ones?):
  - CXR-BERT-specialized embeddings -> FID, KID, Precision/Recall. This is
    the direct text analog of RAD-DINO on the image side: a text encoder
    trained specifically on MIMIC-CXR reports (not generic English), so the
    embedding space actually reflects radiology-report structure.
  - Self-BLEU: does a method repeat itself across different reports (mode
    collapse in text)?
  - Distinct-1/2: raw lexical diversity (unique n-gram ratio).

Q4 -- Intrinsic linguistic quality (is the text well-formed, independent of
      whether it matches real style or the correct diagnosis):
  - Grammar error rate (LanguageTool) -- literal grammatical correctness.
  - Repetition ratio -- catches degenerate/looping generation that grammar
    checkers miss (repeated valid sentences are grammatically fine).
  - Readability (Flesch Reading Ease) distribution vs Real -- not a
    pass/fail score, but flags if a method's reports are suspiciously
    simpler/more convoluted than real reports.
  Perplexity was deliberately left out: no LM exists that's pretrained on
  radiology reports specifically, so any perplexity number would partly
  reflect domain mismatch rather than text quality.

LLM-as-judge -- MedGemma 27B (google/medgemma-27b-text-it), Google's
text-only medical LLM (Gemma 3-based, ~27B params, released 2025). Rates
FLUENCY/COHERENCE/STYLE only -- explicitly instructed not to judge
diagnostic correctness, since that's already covered by CheXpert-Labeler
agreement elsewhere in the pipeline. Requires accepting Health AI Developer
Foundations terms on Hugging Face (gated model) and a logged-in HF token.
Real's reports are judged too (not just the fake methods), so its row in
the summary table is a genuine measured score.

NOTE: this script was written and syntax-checked but NOT execution-tested
(no internet access in the environment it was written in to download any of
these models). Recommend a smoke test with small N values before a full run.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import linalg
from scipy.spatial.distance import cdist
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import textstat
import language_tool_python

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

# =============================================================================
# CONFIG -- edit these directly
# =============================================================================
N_TEXTS = 1000                  # max reports sampled per source, for FID/KID/diversity/linguistic metrics
LLM_JUDGE_SAMPLE_SIZE = 200      # smaller subset for MedGemma judging (expensive per-report)
SEED = 42
KNN_K = 3
SELF_BLEU_SAMPLE_SIZE = 200      # subsample for self-BLEU (full pairwise is O(n^2), too slow at n=1000)
OUTPUT_DIR = Path("./report_similarity_outputs")

REAL_CSV = {
    "path": "",
    "column": "cleaned_report",
}

FAKE_CSVS = [
    {"name": "CXR-IRGen", "path": "",
     "column": "prediction"},
    {"name": "ProgEmu", "path": "",
     "column": None, "headerless": True},
    {"name": "Qwen", "path": "",
     "column": "ClinicalReport"},
]

CXR_BERT_MODEL_ID = "microsoft/BiomedVLP-CXR-BERT-specialized"
MEDGEMMA_MODEL_ID = "google/medgemma-27b-text-it"
USE_4BIT_FOR_MEDGEMMA = True   # 27B in bf16 needs ~54GB; 4-bit brings this down substantially

METHOD_COLORS = {
    "Real": "#7f7f7f",
    "CXR-IRGen": "#2ca02c",
    "ProgEmu": "#1f77b4",
    "Qwen": "#d62728",
}

# =============================================================================


def load_texts_from_csv(csv_path, column, n, seed, headerless=False):
    df = pd.read_csv(csv_path, header=None) if headerless else pd.read_csv(csv_path)
    if column is None:
        column = df.columns[0]
        note = " (headerless CSV, first column)" if headerless else " (using its only/first column)"
        print(f"No column specified for {csv_path} --{note}: '{column}'")
    elif column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}. Columns: {list(df.columns)}")
    texts = df[column].dropna().astype(str).tolist()
    texts = [t for t in texts if t.strip()]
    if not texts:
        raise ValueError(f"No usable text found in column '{column}' of {csv_path}")
    if len(texts) < n:
        print(f"Warning: {csv_path} has only {len(texts)} usable rows, fewer than requested {n}.")
    rng = np.random.RandomState(seed)
    if len(texts) > n:
        idx = rng.choice(len(texts), size=n, replace=False)
        texts = [texts[i] for i in idx]
    return texts


# ---------------------------------------------------------------------------
# Q1a: CXR-BERT embeddings + FID / KID / Precision-Recall
# ---------------------------------------------------------------------------
def build_cxrbert(device):
    tokenizer = AutoTokenizer.from_pretrained(CXR_BERT_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(CXR_BERT_MODEL_ID, trust_remote_code=True).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def extract_cxrbert_embeddings(tokenizer, model, texts, device, batch_size=32, max_length=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="CXR-BERT embeddings"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, add_special_tokens=True, padding=True,
            truncation=True, max_length=max_length, return_tensors="pt",
        ).to(device)
        try:
            emb = model.get_projected_text_embeddings(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask
            )
        except AttributeError:
            # Fallback if the custom method isn't available in this environment's
            # transformers/model revision: mean-pool the last hidden state instead.
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            mask = enc.attention_mask.unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


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
    real_radii = knn_radii(real_feats, k)
    fake_radii = knn_radii(fake_feats, k)
    d_fake_to_real = cdist(fake_feats, real_feats)
    precision = float(np.mean(np.any(d_fake_to_real <= real_radii[None, :], axis=1)))
    d_real_to_fake = cdist(real_feats, fake_feats)
    recall = float(np.mean(np.any(d_real_to_fake <= fake_radii[None, :], axis=1)))
    return precision, recall


# ---------------------------------------------------------------------------
# Q1b: lexical diversity (Self-BLEU, Distinct-n)
# ---------------------------------------------------------------------------
def distinct_n(texts, n):
    ngrams = set()
    total = 0
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i:i + n]))
            total += 1
    return len(ngrams) / max(total, 1)


def self_bleu(texts, sample_size, seed):
    """Average BLEU of each report against a handful of OTHER reports from
    the same set. High self-BLEU = reports are suspiciously similar to each
    other (mode collapse); low = genuinely varied phrasing, as real reports
    naturally are despite radiology's templated style."""
    rng = np.random.RandomState(seed)
    if len(texts) > sample_size:
        idx = rng.choice(len(texts), size=sample_size, replace=False)
        sample = [texts[i] for i in idx]
    else:
        sample = texts
    smoothing = SmoothingFunction().method1
    scores = []
    for i, hyp in enumerate(sample):
        refs = [t.split() for j, t in enumerate(sample) if j != i]
        if not refs:
            continue
        ref_subset = [refs[j] for j in rng.choice(len(refs), size=min(5, len(refs)), replace=False)]
        scores.append(sentence_bleu(ref_subset, hyp.split(), smoothing_function=smoothing))
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# Q4: intrinsic linguistic quality
# ---------------------------------------------------------------------------
def grammar_errors_per_100_words(texts, tool):
    rates = []
    for text in tqdm(texts, desc="Grammar check"):
        n_words = max(len(text.split()), 1)
        n_errors = len(tool.check(text))
        rates.append(100 * n_errors / n_words)
    return rates


def repetition_ratio(text, n=3):
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1 - (len(set(ngrams)) / len(ngrams))


def readability_scores(texts):
    return [textstat.flesch_reading_ease(t) for t in texts]


# ---------------------------------------------------------------------------
# LLM-as-judge: MedGemma 27B, fluency/coherence/style ONLY
# ---------------------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """You are evaluating the WRITING QUALITY of a chest X-ray radiology report.

Judge ONLY fluency, coherence, and whether it reads like a professionally
written radiology report. Do NOT judge whether the described findings are
medically correct or plausible -- that is evaluated separately.

Report:
\"\"\"
{report}
\"\"\"

Rate the writing quality on a scale of 1 (incoherent / not report-like) to
5 (indistinguishable from a professionally written report). Respond with
ONLY a single digit from 1 to 5, nothing else.
"""


def build_medgemma(device, use_4bit):
    tokenizer = AutoTokenizer.from_pretrained(MEDGEMMA_MODEL_ID)
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        MEDGEMMA_MODEL_ID, quantization_config=quant_config,
        torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    return tokenizer, model


@torch.no_grad()
def llm_judge_score(tokenizer, model, text):
    messages = [{"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(report=text)}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    match = re.search(r"[1-5]", response)
    return int(match.group()) if match else None


def run_llm_judge(tokenizer, model, texts_by_source, sample_size, seed):
    """texts_by_source includes 'Real' alongside every fake method -- Real's
    reports get judged too, so its row in the summary table is a genuine
    measured score, not a placeholder."""
    rng = np.random.RandomState(seed)
    scores_by_source = {}
    for name, texts in texts_by_source.items():
        sample = texts if len(texts) <= sample_size else [texts[i] for i in
                                                            rng.choice(len(texts), size=sample_size, replace=False)]
        scores = [llm_judge_score(tokenizer, model, t) for t in tqdm(sample, desc=f"MedGemma judging {name}")]
        scores_by_source[name] = [s for s in scores if s is not None]
    return scores_by_source


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_fid_kid_bars(metrics_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["FID", "KID"]):
        colors = [METHOD_COLORS.get(m, "#999999") for m in metrics_df.index]
        ax.bar(metrics_df.index, metrics_df[col], color=colors)
        ax.set_title(col)
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Report Distributional Realism (CXR-BERT embeddings, lower = closer to real)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(pr_results, output_path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, (precision, recall) in pr_results.items():
        color = METHOD_COLORS.get(name, None)
        ax.scatter(recall, precision, s=180, color=color, edgecolor="black", zorder=5)
        ax.annotate(name, (recall, precision), textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall (diversity)")
    ax.set_ylabel("Precision (fidelity)")
    ax.set_title("Report Fidelity vs. Diversity per Generation Method")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_linguistic_quality(grammar_by_source, repetition_by_source, readability_by_source, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, data, title, ylabel in zip(
        axes,
        [grammar_by_source, repetition_by_source, readability_by_source],
        ["Grammar Errors", "Repetition (3-gram)", "Readability"],
        ["Errors per 100 words", "Fraction repeated 3-grams", "Flesch Reading Ease"],
    ):
        labels = list(data.keys())
        values = [data[l] for l in labels]
        colors = [METHOD_COLORS.get(l, "#999999") for l in labels]
        bp = ax.boxplot(values, patch_artist=True, labels=labels)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Intrinsic Linguistic Quality (independent of realism or diagnostic correctness)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_llm_judge_scores(scores_by_source, output_path):
    labels = list(scores_by_source.keys())
    data = [scores_by_source[l] for l in labels]
    colors = [METHOD_COLORS.get(l, "#999999") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(data, showmedians=True)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 5.5)
    ax.set_ylabel("MedGemma fluency/style score (1-5)")
    ax.set_title("LLM-as-Judge: Writing Quality (MedGemma 27B)\n(fluency/style only, not diagnostic correctness)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_radar(summary_df, output_path):
    """One axis per metric, normalized so the outer edge = best-of-the-
    fakes on that axis (Real excluded, since this compares the fakes to
    each other -- see the FID/KID/Precision-Recall plots for vs-Real)."""
    metrics = list(summary_df.columns)
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    maxes = summary_df.max()
    normalized = summary_df.div(maxes)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for method in normalized.index:
        vals = normalized.loc[method].tolist()
        vals += vals[:1]
        color = METHOD_COLORS.get(method, None)
        ax.plot(angles, vals, color=color, linewidth=2, label=method)
        ax.fill(angles, vals, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.tick_params(axis="x", pad=12)
    ax.set_rlabel_position(200)
    ax.set_title("Report Quality Summary Across All Metrics\n(each axis normalized to the best-performing method)", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sources = {"Real": REAL_CSV}
    for cfg in FAKE_CSVS:
        sources[cfg["name"]] = cfg
    method_names = [cfg["name"] for cfg in FAKE_CSVS]

    texts_by_source = {}
    for name, cfg in sources.items():
        texts = load_texts_from_csv(cfg["path"], cfg["column"], N_TEXTS, SEED, headerless=cfg.get("headerless", False))
        print(f"{name}: {len(texts)} reports from {cfg['path']}")
        texts_by_source[name] = texts

    # ---- Q1a: CXR-BERT FID/KID/Precision-Recall ----
    print("\nLoading CXR-BERT ...")
    cxrbert_tokenizer, cxrbert_model = build_cxrbert(device)
    embeddings_by_source = {
        name: extract_cxrbert_embeddings(cxrbert_tokenizer, cxrbert_model, texts, device)
        for name, texts in texts_by_source.items()
    }
    real_emb = embeddings_by_source["Real"]
    mu_real, sigma_real = real_emb.mean(0), np.cov(real_emb, rowvar=False)

    fid_kid_rows = {}
    pr_results = {}
    for name in method_names:
        fake_emb = embeddings_by_source[name]
        mu_fake, sigma_fake = fake_emb.mean(0), np.cov(fake_emb, rowvar=False)
        fid = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        kid = kernel_inception_distance(real_emb, fake_emb)
        precision, recall = precision_recall(real_emb, fake_emb, k=KNN_K)
        fid_kid_rows[name] = {"FID": fid, "KID": kid}
        pr_results[name] = (precision, recall)

    # Real vs itself is a data-identity case, not a measurement: distance to
    # itself is 0 (FID, KID) and every point trivially matches itself in the
    # manifold check (Precision, Recall = 1). Reported literally as such.
    fid_kid_rows["Real"] = {"FID": 0.0, "KID": 0.0}
    pr_results["Real"] = (1.0, 1.0)

    # ---- Q1b: lexical diversity ----
    print("\nComputing lexical diversity (Self-BLEU, Distinct-n) ...")
    diversity_rows = {}
    for name, texts in texts_by_source.items():
        diversity_rows[name] = {
            "Self-BLEU": self_bleu(texts, SELF_BLEU_SAMPLE_SIZE, SEED),
            "Distinct-1": distinct_n(texts, 1),
            "Distinct-2": distinct_n(texts, 2),
        }

    # ---- Q4: linguistic quality ----
    print("\nRunning grammar check (LanguageTool) ...")
    tool = language_tool_python.LanguageTool("en-US")
    grammar_by_source, repetition_by_source, readability_by_source = {}, {}, {}
    for name, texts in texts_by_source.items():
        grammar_by_source[name] = grammar_errors_per_100_words(texts, tool)
        repetition_by_source[name] = [repetition_ratio(t) for t in texts]
        readability_by_source[name] = readability_scores(texts)

    # ---- LLM-as-judge: MedGemma 27B ----
    print(f"\nLoading MedGemma ({MEDGEMMA_MODEL_ID}, 4-bit={USE_4BIT_FOR_MEDGEMMA}) ...")
    medgemma_tokenizer, medgemma_model = build_medgemma(device, USE_4BIT_FOR_MEDGEMMA)
    judge_scores_by_source = run_llm_judge(medgemma_tokenizer, medgemma_model, texts_by_source,
                                            LLM_JUDGE_SAMPLE_SIZE, SEED)

    # ---- Assemble summary table (Real included, via its split-half baseline for FID/KID/P/R) ----
    all_names = ["Real"] + method_names
    summary_rows = {}
    for name in all_names:
        summary_rows[name] = {
            "FID": fid_kid_rows[name]["FID"],
            "KID": fid_kid_rows[name]["KID"],
            "Precision": pr_results[name][0],
            "Recall": pr_results[name][1],
            "Self-BLEU": diversity_rows[name]["Self-BLEU"],
            "Distinct-1": diversity_rows[name]["Distinct-1"],
            "Distinct-2": diversity_rows[name]["Distinct-2"],
            "Grammar errors/100w": float(np.mean(grammar_by_source[name])),
            "Repetition ratio": float(np.mean(repetition_by_source[name])),
            "Readability (Flesch)": float(np.mean(readability_by_source[name])),
            "LLM judge score (1-5)": float(np.mean(judge_scores_by_source[name]))
            if judge_scores_by_source[name] else float("nan"),
        }
    summary_df = pd.DataFrame(summary_rows).T
    summary_df.to_csv(OUTPUT_DIR / "report_metrics_summary.csv")
    print("\n=== Report metrics summary (Real row: FID/KID/Precision/Recall are 0/0/1/1 by "
          "definition -- Real vs itself -- everything else is Real's true measured score) ===")
    print(summary_df.round(3).to_string())

    # ---- Plots ----
    print("\nGenerating plots ...")
    fid_kid_df = pd.DataFrame({name: fid_kid_rows[name] for name in all_names}).T
    plot_fid_kid_bars(fid_kid_df, OUTPUT_DIR / "report_fid_kid.png")
    plot_precision_recall(pr_results, OUTPUT_DIR / "report_precision_recall.png")
    plot_linguistic_quality(
        {n: grammar_by_source[n] for n in texts_by_source},
        {n: repetition_by_source[n] for n in texts_by_source},
        {n: readability_by_source[n] for n in texts_by_source},
        OUTPUT_DIR / "report_linguistic_quality.png",
    )
    plot_llm_judge_scores(judge_scores_by_source, OUTPUT_DIR / "report_llm_judge_scores.png")

    # For the radar, orient every axis so "higher normalized value = better",
    # matching the FID/KID/Self-BLEU/Grammar/Repetition "lower is better" axes
    # by inverting them (1 / value), so the visual reads consistently.
    radar_df = summary_df.loc[method_names].copy()
    for col in ["FID", "KID", "Self-BLEU", "Grammar errors/100w", "Repetition ratio"]:
        radar_df[col] = 1 / (radar_df[col] + 1e-6)
    plot_summary_radar(radar_df[["FID", "KID", "Precision", "Recall", "Distinct-2",
                                  "Grammar errors/100w", "Repetition ratio", "LLM judge score (1-5)"]],
                        OUTPUT_DIR / "report_summary_radar.png")

    print(f"\nSaved report_metrics_summary.csv and 5 plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()