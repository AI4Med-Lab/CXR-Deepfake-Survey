"""
Image-report matching evaluation using MedSigLIP embeddings.

Picks 1 generated image and 3 candidate reports from 3 different
subfolders (each subfolder = one generation "prompt"), embeds the image
and all 3 reports with MedSigLIP, and picks the report whose text
embedding has the highest cosine similarity to the image embedding as
the model's "guess." This is a zero-shot retrieval-style evaluation
(no generation/parsing involved, unlike the MedGemma prompting approach)
-- it gives a rough score for how distinguishable/faithful your generated
reports are under a CLIP-style embedding model.

The display order of the 3 candidate reports (i.e. which one is "Report
0" vs "Report 1" vs "Report 2") is shuffled per round, so the correct
report is not always at index 0 -- this avoids any positional bias
either in the saved panel or in downstream analysis of predicted_idx.

Can run multiple rounds with different random samples, and saves a
small panel per round showing the image next to all 3 candidate reports
with their similarity scores and which one was chosen vs which one was
correct.

Usage:
    python match_reports_siglip.py \
        --generated-csv /path/to/generated_samples.csv \
        --output-dir /path/to/output_dir \
        --count 3 \
        --rounds 50 \
        --seed 42
"""

import argparse
import random
import textwrap
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/medsiglip-448"

# MedSigLIP (like SigLIP) has a fixed max text sequence length used during
# training; radiology reports can easily run longer than that in raw text.
# We truncate at the tokenizer level (truncation=True below) rather than
# pre-slicing strings, so this is just documented here for clarity.
MAX_TEXT_TOKENS = 64


def pick_samples(generated_csv: Path, count: int, seed: int) -> pd.DataFrame:
    """Pick one row per subfolder, from `count` distinct subfolders.

    Expects generated_csv to have an image_path column and a report or
    prediction column (image_path's parent folder name is the grouping
    key). Row order here defines the "true" image<->report pairing: row 0
    is the image we embed, and row 0's report is the correct answer among
    the `count` candidate reports (rows 0..count-1).
    """
    df = pd.read_csv(generated_csv)
    if "image_path" not in df.columns:
        raise ValueError(f"{generated_csv} must have an 'image_path' column")
    if "report" not in df.columns:
        if "prediction" in df.columns:
            df = df.rename(columns={"prediction": "report"})
        else:
            raise ValueError(f"{generated_csv} must have a 'report' or 'prediction' column")

    df["folder"] = df["image_path"].apply(lambda p: Path(str(p)).parent.name)

    folders = df["folder"].unique().tolist()
    if len(folders) < count:
        raise ValueError(f"Need {count} distinct subfolders, only found {len(folders)}")

    rng = random.Random(seed)
    chosen_folders = rng.sample(folders, count)

    rows = []
    for folder in chosen_folders:
        candidates = df[df["folder"] == folder]
        rows.append(candidates.sample(n=1, random_state=seed).iloc[0])

    return pd.DataFrame(rows).reset_index(drop=True)


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def embed_image_and_reports(
    image_path, reports, processor, model, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed one image and a list of report strings with MedSigLIP.

    Returns:
        (image_embedding, report_embeddings) as L2-normalized tensors of
        shape (embed_dim,) and (len(reports), embed_dim) respectively, so
        that a dot product directly gives cosine similarity.
    """
    image = Image.open(image_path).convert("RGB")

    # Image embedding. SigLIP-style processors expect images and text to
    # go through separate calls (or a combined call); we do them
    # separately here so the number of texts (3 reports) doesn't need to
    # match the number of images (1) the way a naive combined call would
    # assume.
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    text_inputs = processor(
        text=reports,
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # Some checkpoints of this architecture don't expose the
        # `get_image_features` / `get_text_features` convenience methods
        # (they return a bare BaseModelOutputWithPooling instead of a
        # tensor if called that way). Calling the vision/text towers
        # directly and pulling `pooler_output` works regardless, and is
        # what those convenience methods do internally anyway for
        # SigLIP-style models.
        image_out = model.vision_model(**image_inputs)
        text_out = model.text_model(**text_inputs)
        image_embeds = image_out.pooler_output
        text_embeds = text_out.pooler_output

    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    # Squeeze the batch-of-1 image embedding down to a plain vector.
    return image_embeds[0].float().cpu(), text_embeds.float().cpu()


def wrap_text(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def save_panel(
    image_path,
    reports: list[str],
    similarities: list[float],
    predicted_idx: int,
    correct_idx: int,
    output_path: Path,
    image_size: int = 320,
    text_width_chars: int = 60,
) -> None:
    """Image on the left; each candidate report (with its similarity
    score) stacked on the right. The chosen report is boxed green if
    correct, red if wrong; the true report is marked with a star.
    """
    n = len(reports)
    text_panel_width = 620
    row_height = 130
    header_height = 50
    width = image_size + text_panel_width + 40
    height = max(image_size, header_height + row_height * n) + 40

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    try:
        thumb = Image.open(image_path).convert("RGB")
        thumb.thumbnail((image_size, image_size))
        canvas.paste(thumb, (20, 20))
    except Exception as exc:
        draw.text((20, 20 + image_size // 2), f"[failed to load: {exc}]", fill="red", font=font)

    draw.text((image_size + 40, 10), "Candidate reports", fill="black", font=font_bold)

    for i in range(n):
        y = header_height + i * row_height
        x = image_size + 40

        is_predicted = i == predicted_idx
        is_correct_answer = i == correct_idx
        correct_pick = predicted_idx == correct_idx

        label = f"Report {i} (sim={similarities[i]:.4f})"
        if is_correct_answer:
            label += "  [TRUE MATCH]"
        if is_predicted:
            label += "  <- CHOSEN"

        label_color = "black"
        if is_predicted:
            label_color = "green" if correct_pick else "red"

        draw.text((x, y), label, fill=label_color, font=font_bold)
        draw.text(
            (x, y + 22),
            wrap_text(reports[i], text_width_chars),
            fill="black",
            font=font,
        )

        if is_predicted:
            draw.rectangle(
                [x - 5, y - 5, width - 15, y + row_height - 15],
                outline=("green" if correct_pick else "red"),
                width=3,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_round(
    generated_csv: Path,
    output_dir: Path,
    count: int,
    seed: int,
    round_idx: int,
    processor,
    model,
    device,
) -> dict:
    print(f"Round {round_idx} (seed={seed})...")
    samples = pick_samples(generated_csv, count, seed)

    # By construction (see pick_samples), the image at row 0 is the query
    # image and its true matching report is also at row 0 of `samples`.
    # We then shuffle the DISPLAY order of the reports independently of
    # that underlying pairing, so the correct report isn't always shown
    # (and therefore isn't always chosen by chance) at index 0. This
    # mirrors the report-order shuffle used in the MedGemma prompting
    # script, applied here to which "Report i" slot each report lands in.
    image_path = samples.iloc[0]["image_path"]
    true_reports = samples["report"].tolist()
    true_folders = samples["folder"].tolist()

    rng = random.Random(seed)
    display_order = list(range(count))  # display_order[j] = true index shown as "Report j"
    rng.shuffle(display_order)

    reports = [true_reports[true_idx] for true_idx in display_order]
    folders = [true_folders[true_idx] for true_idx in display_order]
    # Row 0 in `samples` is the true match; find where it landed after
    # the shuffle.
    correct_idx = display_order.index(0)

    image_embed, report_embeds = embed_image_and_reports(
        image_path, reports, processor, model, device
    )

    # Cosine similarity of the (normalized) image embedding against each
    # (normalized) report embedding is just their dot product.
    similarities = (report_embeds @ image_embed).tolist()
    predicted_idx = int(max(range(count), key=lambda i: similarities[i]))
    correct = predicted_idx == correct_idx

    result = {
        "round": round_idx,
        "seed": seed,
        "image_path": str(image_path),
        "correct_idx": correct_idx,
        "predicted_idx": predicted_idx,
        "correct": correct,
        **{f"similarity_report_{i}": similarities[i] for i in range(count)},
        **{f"report_{i}": reports[i] for i in range(count)},
        **{f"folder_{i}": folders[i] for i in range(count)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / f"panel_round{round_idx}.png"
    save_panel(image_path, reports, similarities, predicted_idx, correct_idx, panel_path)

    print(f"  Similarities: {[f'{s:.4f}' for s in similarities]}")
    print(f"  Predicted: Report {predicted_idx} | Correct: Report {correct_idx} | "
          f"{'CORRECT' if correct else 'WRONG'}")
    print(f"  Saved: {panel_path}")

    return result


def run(generated_csv: Path, output_dir: Path, count: int, seed: int, rounds: int) -> pd.DataFrame:
    processor, model, device = load_model()

    all_results = []
    for round_idx in range(rounds):
        round_seed = seed + round_idx  # different sample set per round
        result = run_round(
            generated_csv, output_dir, count, round_seed, round_idx, processor, model, device
        )
        all_results.append(result)

    combined = pd.DataFrame(all_results)
    combined_path = output_dir / "match_results_all_rounds.csv"
    combined.to_csv(combined_path, index=False)

    overall_accuracy = combined["correct"].mean()
    print(f"\nOverall accuracy across {rounds} round(s): {overall_accuracy:.2%} "
          f"({combined['correct'].sum()}/{len(combined)})")
    print(f"Combined results saved: {combined_path}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match a generated CXR image to its correct generated report "
        "among distractor reports, using MedSigLIP embedding similarity"
    )
    parser.add_argument(
        "--generated-csv",
        type=Path,
        default=Path("",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(""),
        help="Directory to write per-round panels and combined CSV",
    )
    parser.add_argument("--count", type=int, default=3, help="Number of candidate reports (image is always index 0)")
    parser.add_argument("--rounds", type=int, default=50, help="Number of independent rounds to run")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; round i uses seed + i")
    args = parser.parse_args()

    run(args.generated_csv, args.output_dir, args.count, args.seed, args.rounds)