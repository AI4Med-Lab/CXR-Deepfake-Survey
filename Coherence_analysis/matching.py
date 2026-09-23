"""
Match N generated CXR images to N candidate generated reports using
MedSigLIP embeddings and the Hungarian algorithm for optimal one-to-one
assignment, compared against naive independent per-image argmax.

Expects a CSV with image and report columns. The column names are supplied
on the command line so the script can be used with different metadata files.
"""

import argparse
import random
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/medsiglip-448"
MAX_TEXT_TOKENS = 64


def pick_samples(
    generated_csv: Path, count: int, seed: int, image_column: str, report_column: str
) -> pd.DataFrame:
    """Pick one row per subfolder, or fallback to picking distinct rows if
    subfolders are not diverse/available.
    """
    df = pd.read_csv(generated_csv)

    missing_cols = [c for c in (image_column, report_column) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{generated_csv} must have columns {missing_cols} -- "
            f"rename your columns before running, found: {list(df.columns)}"
        )

    # Drop rows with missing/empty reports before sampling -- a NaN or blank
    # report crashes the tokenizer downstream (expects str, gets float('nan')),
    # and silently including it would make sampling behavior seed-dependent
    # on which rows happen to be invalid.
    n_before = len(df)
    df = df[df[report_column].notna()]
    df = df[df[report_column].astype(str).str.strip() != ""]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[warn] Dropped {n_dropped}/{n_before} rows with missing/empty reports before sampling.")

    # Extract parent folder name
    df["folder"] = df[image_column].apply(lambda p: Path(str(p)).parent.name)
    folders = df["folder"].unique().tolist()

    # Fallback: if all images share the same folder, treat each row as a distinct group
    if len(folders) < count:
        df["folder"] = [f"row_{i}" for i in range(len(df))]
        folders = df["folder"].unique().tolist()

    if len(folders) < count:
        raise ValueError(
            f"Need {count} distinct samples, but dataset only has {len(folders)} usable rows."
        )

    rng = random.Random(seed)
    chosen_folders = rng.sample(folders, count)

    rows = []
    for folder in chosen_folders:
        candidates = df[df["folder"] == folder]
        rows.append(candidates.sample(n=1, random_state=seed).iloc[0])

    return pd.DataFrame(rows).reset_index(drop=True)


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def embed_images_and_reports(
    image_paths, reports, processor, model, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed a list of images and a list of report strings with MedSigLIP."""
    images = [Image.open(p).convert("RGB") for p in image_paths]

    image_inputs = processor(images=images, return_tensors="pt").to(device)
    text_inputs = processor(
        text=reports,
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        image_out = model.vision_model(**image_inputs)
        text_out = model.text_model(**text_inputs)
        image_embeds = image_out.pooler_output
        text_embeds = text_out.pooler_output

    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    return image_embeds.float().cpu(), text_embeds.float().cpu()


def solve_assignment(similarity: np.ndarray) -> list[int]:
    """Optimal one-to-one assignment via Hungarian algorithm."""
    row_idx, col_idx = linear_sum_assignment(-similarity)
    assignment = [0] * similarity.shape[0]
    for i, j in zip(row_idx, col_idx):
        assignment[i] = int(j)
    return assignment


def solve_greedy_independent(similarity: np.ndarray) -> tuple[list[int], bool]:
    """Naive per-image argmax."""
    assignment = similarity.argmax(axis=1).tolist()
    has_conflict = len(set(assignment)) < len(assignment)
    return assignment, has_conflict


def wrap_text(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def save_panel(
    image_paths,
    reports: list[str],
    similarity: np.ndarray,
    hungarian_assignment: list[int],
    correct_assignment: list[int],
    output_path: Path,
    thumb_size: int = 220,
    text_width_chars: int = 46,
) -> None:
    n = len(image_paths)
    text_panel_width = 480
    sims_panel_width = 260
    row_height = thumb_size + 20
    header_height = 40
    width = thumb_size + text_panel_width + sims_panel_width + 60
    height = header_height + row_height * n

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    draw.text((10, 10), "Image", fill="black", font=font_bold)
    draw.text(
        (thumb_size + 20, 10),
        "Assigned report (Hungarian)",
        fill="black",
        font=font_bold,
    )
    draw.text(
        (thumb_size + text_panel_width + 30, 10),
        "Similarities (all reports)",
        fill="black",
        font=font_bold,
    )

    for i in range(n):
        y = header_height + i * row_height
        image_path = image_paths[i]
        assigned_j = hungarian_assignment[i]
        correct = assigned_j == correct_assignment[i]

        try:
            thumb = Image.open(image_path).convert("RGB")
            thumb.thumbnail((thumb_size, thumb_size))
            canvas.paste(thumb, (10, y))
        except Exception as exc:
            draw.text(
                (10, y + thumb_size // 2), f"[failed to load: {exc}]", fill="red", font=font
            )

        border_color = "green" if correct else "red"
        draw.rectangle(
            [5, y - 5, thumb_size + 15, y + thumb_size + 5],
            outline=border_color,
            width=3,
        )

        header = f"-> Report {assigned_j} (sim={similarity[i, assigned_j]:.4f})"
        draw.text((thumb_size + 20, y), header, fill=border_color, font=font_bold)
        draw.text(
            (thumb_size + 20, y + 20),
            wrap_text(reports[assigned_j], text_width_chars),
            fill="black",
            font=font,
        )

        sims_x = thumb_size + text_panel_width + 30
        for j in range(similarity.shape[1]):
            marker = " *" if j == assigned_j else ""
            marker += " (true)" if j == correct_assignment[i] else ""
            draw.text(
                (sims_x, y + j * 18),
                f"Report {j}: {similarity[i, j]:.4f}{marker}",
                fill="black",
                font=font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_round(
    generated_csv: Path,
    output_dir: Path,
    count: int,
    seed: int,
    round_idx: int,
    image_column: str,
    report_column: str,
    processor,
    model,
    device,
) -> dict:
    print(f"Round {round_idx} (seed={seed})...")
    samples = pick_samples(generated_csv, count, seed, image_column, report_column)

    image_paths = samples[image_column].tolist()
    true_reports = samples[report_column].tolist()
    true_folders = samples["folder"].tolist()

    rng = random.Random(seed)
    display_order = list(range(count))
    rng.shuffle(display_order)

    reports = [true_reports[true_idx] for true_idx in display_order]
    folders = [true_folders[true_idx] for true_idx in display_order]
    true_to_display = {true_idx: j for j, true_idx in enumerate(display_order)}
    correct_assignment = [true_to_display[i] for i in range(count)]

    image_embeds, report_embeds = embed_images_and_reports(
        image_paths, reports, processor, model, device
    )

    similarity = (image_embeds @ report_embeds.T).numpy()

    hungarian_assignment = solve_assignment(similarity)
    greedy_assignment, has_conflict = solve_greedy_independent(similarity)

    hungarian_correct = sum(
        1 for i in range(count) if hungarian_assignment[i] == correct_assignment[i]
    )
    greedy_correct = sum(
        1 for i in range(count) if greedy_assignment[i] == correct_assignment[i]
    )
    assignments_differ = hungarian_assignment != greedy_assignment

    result = {
        "round": round_idx,
        "seed": seed,
        "n": count,
        "hungarian_correct_count": hungarian_correct,
        "hungarian_accuracy": hungarian_correct / count,
        "greedy_correct_count": greedy_correct,
        "greedy_accuracy": greedy_correct / count,
        "greedy_had_conflict": has_conflict,
        "hungarian_vs_greedy_differ": assignments_differ,
        **{f"image_{i}_path": image_paths[i] for i in range(count)},
        **{f"image_{i}_correct_report": correct_assignment[i] for i in range(count)},
        **{f"image_{i}_hungarian_report": hungarian_assignment[i] for i in range(count)},
        **{f"image_{i}_greedy_report": greedy_assignment[i] for i in range(count)},
        **{f"report_{j}_text": reports[j] for j in range(count)},
        **{f"report_{j}_folder": folders[j] for j in range(count)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / f"panel_round{round_idx}.png"
    save_panel(
        image_paths,
        reports,
        similarity,
        hungarian_assignment,
        correct_assignment,
        panel_path,
    )

    print(
        f"   Similarity matrix (rows=images, cols=reports):\n{np.round(similarity, 4)}"
    )
    print(
        f"   Hungarian assignment: {hungarian_assignment}"
        f" ({hungarian_correct}/{count} correct)"
    )
    print(
        f"   Greedy assignment:    {greedy_assignment}"
        f" ({greedy_correct}/{count} correct){'   [CONFLICT]' if has_conflict else ''}"
    )
    print(f"   Saved: {panel_path}")

    return result


def run(
    generated_csv: Path,
    output_dir: Path,
    count: int,
    seed: int,
    rounds: int,
    image_column: str,
    report_column: str,
) -> pd.DataFrame:
    processor, model, device = load_model()

    all_results = []
    for round_idx in range(rounds):
        round_seed = seed + round_idx
        result = run_round(
            generated_csv,
            output_dir,
            count,
            round_seed,
            round_idx,
            image_column,
            report_column,
            processor,
            model,
            device,
        )
        all_results.append(result)

    combined = pd.DataFrame(all_results)
    combined_path = output_dir / "match_results_all_rounds.csv"
    combined.to_csv(combined_path, index=False)

    hungarian_overall = combined["hungarian_correct_count"].sum() / combined["n"].sum()
    greedy_overall = combined["greedy_correct_count"].sum() / combined["n"].sum()
    conflict_rate = combined["greedy_had_conflict"].mean()
    disagreement_rate = combined["hungarian_vs_greedy_differ"].mean()

    print(f"\nOverall Hungarian (optimal) accuracy: {hungarian_overall:.2%}")
    print(f"Overall greedy (independent argmax) accuracy: {greedy_overall:.2%}")
    print(
        f"Rounds where greedy produced a conflicting (invalid) assignment:"
        f" {conflict_rate:.2%}"
    )
    print(
        f"Rounds where Hungarian and greedy assignments differed:"
        f" {disagreement_rate:.2%}"
    )
    print(f"Combined results saved: {combined_path}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Match N generated CXR images to N candidate generated reports using"
            " MedSigLIP embeddings and the Hungarian algorithm for optimal"
            " one-to-one assignment"
        )
    )
    parser.add_argument(
        "--generated-csv",
        type=Path,
        required=True,
        help="Path to a CSV with 'image_path' and 'report' columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-round panels and combined CSV",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of images/reports to sample per round (N x N)",
    )
    parser.add_argument(
        "--rounds", type=int, default=50, help="Number of independent rounds to run"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base seed; round i uses seed + i"
    )
    parser.add_argument(
        "--image-column", default="image_path", help="CSV column containing image paths"
    )
    parser.add_argument(
        "--report-column", default="report", help="CSV column containing reports"
    )
    args = parser.parse_args()
    run(
        args.generated_csv,
        args.output_dir,
        args.count,
        args.seed,
        args.rounds,
        args.image_column,
        args.report_column,
    )